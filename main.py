# ─────────────────────────── IMPORTS ───────────────────────────
import os, re, json, itertools, sys
from io import BytesIO
from typing import List, Dict
import openai
import numpy as np
from dotenv import load_dotenv
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ───────────────────── CONFIG & PATHS ─────────────────────
load_dotenv()
TEST_MODE = False
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
eleven = ElevenLabs(api_key=eleven_api_key)

IN_VIDEO = "input_video.mov"
AUDIO_ORIG = "extracted_audio.mp3"
SRT_RAW = "raw_transcript.srt"
TXT_CLEAN = "clean_transcript.txt"
AUDIO_OUT = "final_aligned_audio.mp3"
VIDEO_OUT = "final_video_with_new_audio.mp4"
CHUNKS_DIR = "audio_chunks"

# ───────────────────── SRT HELPERS ─────────────────────
_W_RE = re.compile(r"\w+")
def _ts_to_sec(ts:str)->float:
    h,m,sms = ts.split(":"); s,ms = sms.split(",")
    return int(h)*3600+int(m)*60+int(s)+int(ms)/1000
def parse_srt(path)->List[Dict]:
    patt = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s+([\s\S]+?)(?=\n{2,}|\Z)", re.MULTILINE)
    segs=[]
    for idx,st,txt in patt.findall(open(path,encoding="utf-8").read().strip()):
        segs.append({"idx":int(idx),"start":_ts_to_sec(st),"text":re.sub(r"\s+"," ",txt).strip()})
    return segs

# ───── GPT-4 MAPPING (caption index only) ─────
def ask_mapping(chunks,srt,feedback=None):
    total=len(chunks)
    srt_block = "\n".join(f"{s['idx']}: {s['text']}" for s in srt)
    chunk_block = "\n".join(f"{i+1}: {t}" for i,t in enumerate(chunks))
    # --- IMPROVED PROMPT ---
    sys = (
        "You are an expert audio-visual alignment specialist. Your task is to synchronize a clean, rewritten script (CHUNKS) with a raw, timed transcript (SRT).\n"
        "Your goal is to find the best starting point in the original timeline for each clean chunk.\n\n"
        f"Follow this process for EACH chunk from 1 to {total}:\n"
        "1. Read the chunk and identify its main subject or the key information it introduces.\n"
        "2. Scan the SRT captions starting from SRT caption #1.\n"
        "3. Find the VERY FIRST SRT caption that introduces the same subject or key information. This is your match.\n"
        "4. The chosen `caption_idx` for each chunk should generally be greater than or equal to the `caption_idx` of the previous chunk.\n\n"
        "Return ONLY a JSON array of objects. Each object must have these keys:\n"
        '  "chunk": integer (1-based, from the input)\n'
        '  "caption_idx": integer (the index of the matched SRT caption)\n\n'
        "EXAMPLE:\n"
        "If SRT is:\n"
        "1: Hello and welcome.\n"
        "2: Today we discuss apples.\n"
        "3. They are red or green.\n"
        "And CHUNKS are:\n"
        "1: Welcome to the show.\n"
        "2: Let's talk about apples, which come in various colors.\n\n"
        "The correct output would be:\n"
        '[\n  {"chunk": 1, "caption_idx": 1},\n  {"chunk": 2, "caption_idx": 2}\n]'
    )

    if feedback:
        sys += f"\n\nPrevious output was invalid: {feedback}\nPlease correct."
    user = f"SRT:\n{srt_block}\n\nCHUNKS:\n{chunk_block}"
    rsp = openai.chat.completions.create(
        model="gpt-4.1", # Using the specified model
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        max_tokens=4096,
        temperature=0.2,
        response_format={"type": "json_object"} # Force JSON output
    )
    return rsp.choices[0].message.content

def map_chunks_with_llm(chunks,srt):
    attempts=0; feedback=None
    while attempts<2:
        print(f"Attempting LLM mapping (Attempt {attempts + 1}/2)...")
        raw=ask_mapping(chunks,srt,feedback)
        try:
            # When using json_object, the response is already a JSON string
            data=json.loads(raw)
            # The AI might wrap the array in a parent key, let's handle that
            if isinstance(data, dict):
                # Find the key that holds the list
                list_key = next((key for key, value in data.items() if isinstance(value, list)), None)
                if list_key:
                    data = data[list_key]
                else:
                    raise ValueError("JSON object returned, but no list found within it.")
            if not isinstance(data, list):
                raise TypeError(f"Expected a list, but got {type(data)}.")
            if len(data)!=len(chunks):
                raise ValueError(f"Expected {len(chunks)} items, got {len(data)}.")
            print("LLM mapping successful.")
            return data
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f" -> LLM mapping failed on attempt {attempts + 1}.")
            feedback=f"Parsing error: {e}\nRaw response from AI:\n---\n{raw[:1000]}\n---"
            attempts+=1
            if attempts < 2:
                print(" Retrying...")
    raise ValueError("GPT-4 mapping failed twice. Check the logged raw response from the AI.")

# ───── TIMELINE BUILD ─────
def build_timeline(mapping, srt, audio_chunks, video_duration_sec, fudge=0.5):
    """
    Builds a timeline by placing audio chunks based on the AI map, then
    proportionally scaling the pauses between them to fit the video's duration.
    """
    print("Building proportionally scaled timeline...")
    # Ensure mapping is processed in chronological chunk order
    mapping = sorted(mapping, key=lambda x: x['chunk'])
    srtD = {s['idx']: s for s in srt}
    
    # --- 1. Build an ideal timeline with variable pauses based on AI map ---
    ideal_timeline = []
    last_end_time = 0
    total_audio_duration_ms = 0
    for m in mapping:
        chunk_index = m['chunk'] - 1
        if chunk_index < len(audio_chunks) and audio_chunks[chunk_index]:
            chunk = audio_chunks[chunk_index]
            chunk_duration = len(chunk)
            total_audio_duration_ms += chunk_duration
            cap = srtD.get(m['caption_idx'], srt[0])
            offset_words = words_before_punct(cap["text"])
            proposed_start = (cap["start"] + offset_words * fudge) * 1000
            start_time = max(proposed_start, last_end_time)
            end_time = start_time + chunk_duration
            ideal_timeline.append({"start": start_time, "end": end_time, "chunk_index": chunk_index})
            last_end_time = end_time
            
    if not ideal_timeline: return []

    # --- 2. Calculate ideal pauses and total duration ---
    ideal_pauses = []
    for i in range(len(ideal_timeline) - 1):
        pause = ideal_timeline[i+1]['start'] - ideal_timeline[i]['end']
        ideal_pauses.append(pause)
    
    total_ideal_duration = ideal_timeline[-1]['end'] if ideal_timeline else 0
    video_duration_ms = video_duration_sec * 1000
    overshoot_ms = total_ideal_duration - video_duration_ms

    # --- 3. Smartly compress pauses if there's an overshoot ---
    if overshoot_ms > 0:
        print(f"Warning: Ideal timeline ({total_ideal_duration}ms) exceeds video duration ({video_duration_ms}ms). Compressing pauses.")
        MINI_PAUSE_MS = 200 # The minimum pause before the last chunk
        
        # Step 1: Try to compensate by shrinking the last pause first
        last_pause_original = ideal_pauses[-1] if ideal_pauses else 0
        last_pause_reduction = min(overshoot_ms, max(0, last_pause_original - MINI_PAUSE_MS))
        
        ideal_pauses[-1] = last_pause_original - last_pause_reduction
        remaining_overshoot = overshoot_ms - last_pause_reduction
        
        # Step 2: If still over, proportionally shrink the rest of the pauses
        if remaining_overshoot > 0:
            remaining_pauses_total = sum(ideal_pauses[:-1])
            if remaining_pauses_total > 0:
                scaling_factor = max(0, (remaining_pauses_total - remaining_overshoot) / remaining_pauses_total)
                print(f"Scaling remaining pauses by a factor of {scaling_factor:.2f} to fit.")
                for i in range(len(ideal_pauses) - 1):
                    ideal_pauses[i] *= scaling_factor
            else:
                print("Warning: No remaining pauses to shrink. Final audio may still be too long.")

    # --- 4. Build the final, scaled timeline ---
    final_timeline = []
    current_time_ms = 0
    for i, item in enumerate(ideal_timeline):
        chunk_duration = item['end'] - item['start']
        final_start = current_time_ms
        final_end = final_start + chunk_duration
        final_timeline.append({"start": final_start, "end": final_end, "chunk_index": item['chunk_index']})
        
        if i < len(ideal_pauses):
            current_time_ms = final_end + ideal_pauses[i]
            
    return final_timeline

def words_before_punct(caption:str)->int:
    """Count words in caption BEFORE first '.', '!' or '?'."""
    m=re.search(r'[.!?]',caption)
    if not m: return 0
    prefix=caption[:m.start()]
    return len(_W_RE.findall(prefix))

def create_master(audio_chunks, timeline, vlen, out):
    """Creates the final audio track by overlaying chunks according to the timeline."""
    bed = AudioSegment.silent(int(vlen * 1000 + 500))
    for t in timeline:
        chunk_index = t['chunk_index']
        if chunk_index < len(audio_chunks) and audio_chunks[chunk_index]:
            seg = audio_chunks[chunk_index]
            bed = bed.overlay(seg, position=int(t['start']))
    bed.export(out, format="mp3", bitrate="192k")

# ───── SMART TRIM & MUX ─────
def smart_trim_and_mux(timeline, video_path, audio_path, final_video_out):
    print("\n--- Starting Smart Trim and Final Render Process ---")
    video = VideoFileClip(video_path)
    audio = AudioSegment.from_mp3(audio_path)
    fps = video.fps
    STATIC_THRESHOLD = 1.0  # Threshold for Mean Squared Error to detect static frames.

    pauses = []
    for i in range(len(timeline) - 1):
        pause_start_sec = timeline[i]['end'] / 1000.0
        pause_end_sec = timeline[i+1]['start'] / 1000.0
        duration = pause_end_sec - pause_start_sec
        if duration > 0.5:
            pauses.append({"start": pause_start_sec, "end": pause_end_sec, "duration": duration})
    
    cuts_to_make_sec = []
    if pauses:
        for pause in pauses:
            analysis_start = pause['start'] + 0.5
            frames_to_check = np.arange(analysis_start, pause['end'], 8.0/fps)
            if len(frames_to_check) < 3: continue

            static_streak = 0
            for i in range(len(frames_to_check) - 1):
                frame1 = video.get_frame(frames_to_check[i])
                frame2 = video.get_frame(frames_to_check[i+1])
                
                # Calculate Mean Squared Error instead of perfect equality
                mse = np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)
                
                if mse < STATIC_THRESHOLD:
                    static_streak += 1
                else:
                    static_streak = 0
                
                if static_streak >= 3:
                    cut_start = frames_to_check[i - 2]
                    cut_end = frames_to_check[i+1]
                    cuts_to_make_sec.append((cut_start, cut_end))
                    static_streak = 0
    
    if not cuts_to_make_sec:
        print("No static segments found to trim. Merging directly.")
        final_clip = video.with_audio(AudioFileClip(audio_path))
        final_clip.write_videofile(final_video_out, codec="libx264", audio_codec="aac")
        video.close()
        return

    print(f"Identified {len(cuts_to_make_sec)} static segments to trim. Performing single render...")
    
    clips_to_keep = []
    last_cut_end = 0
    for start, end in sorted(cuts_to_make_sec):
        clips_to_keep.append((last_cut_end, start))
        last_cut_end = end
    clips_to_keep.append((last_cut_end, video.duration))

    # Build trimmed video and audio in memory
    final_video_clips = [video.subclip(s, e) for s, e in clips_to_keep if s < e]
    trimmed_video = concatenate_videoclips(final_video_clips)
    
    final_audio_segments = [audio[int(s*1000):int(e*1000)] for s, e in clips_to_keep if s < e]
    trimmed_audio = sum(final_audio_segments)

    # Export trimmed audio to a temporary file for moviepy
    temp_audio_path = "temp_trimmed_for_mux.mp3"
    trimmed_audio.export(temp_audio_path, format="mp3")
    
    # Set the new audio and render the final video in one go
    final_clip = trimmed_video.with_audio(AudioFileClip(temp_audio_path))
    final_clip.write_videofile(final_video_out, codec="libx264", audio_codec="aac")
    
    # Clean up
    os.remove(temp_audio_path)
    video.close()
    print("Smart trimming and final render complete.")


# ───── OTHER STEPS ─────
def extract_audio(v,a): VideoFileClip(v).audio.write_audiofile(a,codec="mp3",bitrate="192k")
def transcribe(a,s): open(s,"w", encoding="utf-8").write(openai.audio.transcriptions.create(model="whisper-1",file=open(a,"rb"),response_format="srt"))
def clean(src,dst):
    txt=open(src, encoding="utf-8").read()
    rsp=openai.chat.completions.create(model="gpt-4.1", # Using the specified model
        messages=[{"role": "system", "content": (
            "You are a transcript editor. Clean the following SRT content by removing all timestamps and index numbers, "
            "fixing any grammar or spelling errors, and ensuring a natural, readable flow. The output should be a single block of plain text, with paragraphs separated by a blank line. "
            "Crucially, ensure the following proper nouns are spelled and cased correctly throughout the text: "
            "However, your cleaned transcript should remain close to the original, only making changes necessary for clarity, grammar, and readability."
        )},
                  {"role":"user","content":txt}],temperature=0.6)
    open(dst,"w", encoding="utf-8").write(rsp.choices[0].message.content)
def tts(clean_fp):
    sents=[s.strip() for s in re.split(r'(?<=[.!?])\s+',open(clean_fp, encoding="utf-8").read()) if s.strip()]
    aud=[]
    # In TEST_MODE, we just load existing chunks.
    if TEST_MODE:
        print("--- TEST MODE: Loading existing audio chunks ---")
        for i in range(1,len(sents)+1):
            p=os.path.join(CHUNKS_DIR,f"chunk_{i}.mp3")
            if os.path.exists(p):
                aud.append(AudioSegment.from_mp3(p))
            else:
                print(f"Warning: Chunk file not found: {p}. Appending None.")
                aud.append(None)
        return aud,sents
    # This part only runs if TEST_MODE is False
    print("--- FULL MODE: Generating new audio chunks ---")
    os.makedirs(CHUNKS_DIR,exist_ok=True)
    for i,txt in enumerate(sents,1):
        try:
            data=b"".join(eleven.text_to_speech.convert(
                voice_id="UgBBYS2sOqTuMpoF3BR0",text=txt,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(speed=0.85)))
            open(os.path.join(CHUNKS_DIR,f"chunk_{i}.mp3"),"wb").write(data)
            aud.append(AudioSegment.from_mp3(BytesIO(data)))
        except: aud.append(None)
    return aud,sents

# ───── MAIN ─────
def main():
    if not os.path.exists(IN_VIDEO): return print("Video not found.")
    # --- Pre-processing steps: Only run if not in TEST_MODE ---
    if not TEST_MODE:
        print("--- FULL MODE: Running pre-processing steps ---")
        extract_audio(IN_VIDEO,AUDIO_ORIG)
        transcribe(AUDIO_ORIG,SRT_RAW)
        clean(SRT_RAW,TXT_CLEAN)
    else:
        print("--- TEST MODE: Skipping pre-processing and audio generation ---")
    
    # --- Manual Review Step ---
    if not os.path.exists(TXT_CLEAN):
        print(f"Error: Clean transcript '{TXT_CLEAN}' not found. Please run with TEST_MODE=False first.")
        return
        
    print(f"\n--- MANUAL REVIEW ---")
    print(f"Clean transcript created at '{TXT_CLEAN}'.")
    print("Please open the file, review it, and make any necessary edits.")
    
    while True:
        action = input("Type 'go' to proceed with audio generation, or 'stop' to exit: ").lower()
        if action == 'go':
            print("Proceeding with script...")
            break
        elif action == 'stop':
            print("Exiting script as requested.")
            sys.exit()
        else:
            print("Invalid input. Please type 'go' or 'stop'.")

    # --- Core Logic: Runs in both modes ---
    # `tts` function will load existing chunks in TEST_MODE
    audio_chunks,chunks=tts(TXT_CLEAN)
    # These files must exist for TEST_MODE to work
    if not os.path.exists(SRT_RAW) or not os.path.exists(TXT_CLEAN):
        print(f"Error: In TEST_MODE, '{SRT_RAW}' and '{TXT_CLEAN}' must exist. Run with TEST_MODE=False first.")
        return
    srt=parse_srt(SRT_RAW)
    mapping=map_chunks_with_llm(chunks,srt)
    vlen=VideoFileClip(IN_VIDEO).duration
    # Build the timeline using the new proportional pause logic
    tl=build_timeline(mapping,srt,audio_chunks,vlen)
    create_master(audio_chunks,tl,vlen,AUDIO_OUT)
    
    # --- Smart Trim and Final Muxing Step ---
    if not TEST_MODE:
        smart_trim_and_mux(tl, IN_VIDEO, AUDIO_OUT, VIDEO_OUT)
    
    print("✔︎ Finished")

if __name__=="__main__":
    main()
