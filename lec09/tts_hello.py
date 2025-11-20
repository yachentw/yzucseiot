import argparse
from gtts import gTTS
import subprocess
import tempfile

# Set up argument parser with default values
parser = argparse.ArgumentParser(description="Generate speech and play it.")
parser.add_argument('text', type=str, nargs='?', default="hello", help="Text to convert to speech")
parser.add_argument('lang', type=str, nargs='?', default="en", help="Language for the speech (e.g., 'en' for English)")

# Parse arguments
args = parser.parse_args()

# Generate temporary file with delete=True
with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
    temp_path = temp_file.name
    tts = gTTS(text=args.text, lang=args.lang)
    tts.save(temp_path)

    # Play the MP3 file
    try:
        subprocess.run(
            ['vlc', '--play-and-exit', temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("VLC is not installed or not found in the PATH.")
