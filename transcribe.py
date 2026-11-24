from faster_whisper import WhisperModel 
from pathlib import Path 

# Locate the downloaded audio file 
audio_dir = Path("audio")
audio_files = list(audio_dir.glob("*.mp3"))

if not audio_files:
    raise FileNotFoundError("No MP3 file found in the audio folder")

audio_path = audio_files[0]

# Load a lightweight Whisper model 
model = WhisperModel("large-v3", device = "cpu", compute_type = "int8")

# Transcribe the audio
segments, info = model.transcribe(
    str(audio_path),
    language = None,
    beam_size = 5,
    condition_on_previous_text = False,
    vad_filter = True,
    word_timestamps = True,
    clip_timestamps = "0,60"
)

print("Detected language:", info.language)

# Save the transcript with timestamps
output_path = Path("transcript.txt")

words = []
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
        
    for word in segment.words:
        words.append({"word": word.word, "start": word.start, "end": word.end})
        
print(len(words))
print(words[0])
print(words[-1])
            

print(f"\nTranscript saved to: {output_path.resolve()}")
print(f"Detected language: {info.language}")