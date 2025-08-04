Of course\! Here is an easy-to-understand README file for your Python script.

-----

# AI Video Dubbing & Pacing Script

This Python script automates the process of replacing a video's original audio with a new, high-quality voiceover. It uses AI to transcribe, clean, and re-voice the narration, then intelligently synchronizes the new audio to the video.

As a final step, it "smart-trims" the video by removing long, static pauses, resulting in a more polished and engaging final product.

-----

## 🚀 Features

  * **Audio Extraction**: Automatically extracts the audio from the source video.
  * **AI Transcription**: Uses OpenAI's **Whisper** to generate a timed transcript (`.srt`).
  * **AI Script Cleaning**: Uses **GPT-4** to correct grammar, improve flow, and format the transcript into a clean script.
  * **Manual Review Step**: Pauses execution to allow you to review and edit the cleaned script before generating audio.
  * **High-Quality Voice Generation**: Uses **ElevenLabs** to generate a natural, high-quality voiceover from the final script.
  * **Intelligent Timeline Sync**: Uses **GPT-4** to map the new audio sentences to the original video's timing, ensuring the new narration aligns with the on-screen content.
  * **Smart Video Trimming**: Analyzes the video for visually static or frozen moments during pauses in the narration and automatically cuts them out.
  * **Test Mode**: Includes a `TEST_MODE` to skip expensive API calls and re-use existing files, which is great for debugging the timeline and trimming logic.

-----

## ⚙️ How It Works: The Workflow

The script follows a clear, step-by-step process to transform your video:

1.  **Extract & Transcribe** 🎧→📝

      * The audio from `input_video.mov` is extracted.
      * Whisper transcribes this audio into a raw `.srt` file with precise timestamps.

2.  **Clean & Review** ✨→🧑‍💻

      * GPT-4 reads the raw `.srt` file, removes timestamps, fixes errors, and produces a clean, readable `clean_transcript.txt`.
      * The script pauses and asks for your approval. You can open `clean_transcript.txt` and make any final manual edits.

3.  **Generate & Sync** 🎙️→🧠

      * Once you approve, the script sends the cleaned text to ElevenLabs, which generates a new audio voiceover, sentence by sentence.
      * GPT-4 then compares the new sentences with the original `.srt` transcript to find the best start time for each new audio chunk in the original video's timeline.

4.  **Build & Trim** ✂️→🎬

      * A new, complete audio track is built by placing the voiceover chunks onto a silent track according to the timeline. Pauses between sentences are proportionally scaled to fit the total video length.
      * The script analyzes the original video for static frames during these new, silent pauses. It then trims these static sections from the video.
      * Finally, the trimmed video is merged (muxed) with the new audio track to create the final output file.

-----

## 🛠️ Setup & Installation

Before running the script, you need to set up your environment.

### 1\. Prerequisites

You need Python 3 installed, along with the following libraries. You can install them all using pip:

```bash
pip install openai moviepy pydub elevenlabs-sdk python-dotenv numpy
```

### 2\. API Keys

The script requires API keys for two services:

  * **OpenAI** (for Whisper and GPT-4)
  * **ElevenLabs** (for Text-to-Speech)

### 3\. Configuration

1.  Create a file named `.env` in the same directory as the script.

2.  Add your API keys to this file like this:

    ```
    OPENAI_API_KEY="sk-your-openai-api-key"
    ELEVENLABS_API_KEY="your-elevenlabs-api-key"
    ```

-----

## ▶️ How to Use

1.  **Place Your Video**: Put your source video in the project directory and name it `input_video.mov`.
2.  **Set the Mode**:
      * **Full Run**: To run the entire process from scratch, set `TEST_MODE = False` at the top of the script. This will use the APIs and can incur costs.
      * **Test Run**: To quickly test the timeline or trimming logic without re-generating files, set `TEST_MODE = True`. This will skip audio extraction, transcription, and voice generation, and instead use the files that already exist from a previous run.
3.  **Run the Script**: Execute the script from your terminal:
    ```bash
    python your_script_name.py
    ```
4.  **Manual Review**: The script will stop and wait for your input after creating `clean_transcript.txt`. Open this file, check it for accuracy, and save any changes. Then, return to the terminal and type `go` to continue.

-----

## 📂 File Structure

The script creates several files and folders. Here's what they are:

  * `input_video.mov`: **(You provide this)** The original video file.
  * `extracted_audio.mp3`: The original audio extracted from the video.
  * `raw_transcript.srt`: The raw, timed transcript from Whisper.
  * `clean_transcript.txt`: The cleaned-up script for your review and for TTS generation.
  * `/audio_chunks/`: A folder containing the individual audio files for each sentence of the new voiceover.
  * `final_aligned_audio.mp3`: The complete, final audio track with new narration and correctly timed pauses.
  * `final_video_with_new_audio.mp4`: **The final output\!** Your trimmed video with the new high-quality audio.