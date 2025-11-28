import argparse
from gtts import gTTS
import tempfile
import pygame  # 引入 pygame
import time
import os

parser = argparse.ArgumentParser(description="Generate speech and play it.")
parser.add_argument('text', type=str, nargs='?', default="hello", help="Text to convert to speech")
parser.add_argument('lang', type=str, nargs='?', default="en", help="Language for the speech")

args = parser.parse_args()

# 初始化 pygame 音效模組 (不初始化圖形介面，適合 SSH/Headless)
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Error initializing audio: {e}")
    exit(1)

with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
    # 注意：pygame 在 Windows/Linux 有時對開啟中的檔案會有鎖定問題
    # 所以這裡我們設定 delete=False，手動清理
    temp_path = temp_file.name
    
    print(f"Generating audio for: '{args.text}'...")
    try:
        tts = gTTS(text=args.text, lang=args.lang)
        tts.save(temp_path)
    except Exception as e:
        print(f"Error generating TTS: {e}")
        os.remove(temp_path)
        exit(1)

    print("Playing audio with pygame...")
    try:
        # 載入音樂
        pygame.mixer.music.load(temp_path)
        # 播放
        pygame.mixer.music.play()

        # 因為 pygame 播放是非同步的 (不會卡住程式)，
        # 我們需要寫一個迴圈等待播放結束
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except pygame.error as e:
        print(f"Pygame playback error: {e}")
    finally:
        # 釋放資源並刪除暫存檔
        pygame.mixer.quit()
        try:
            os.remove(temp_path)
        except OSError:
            pass
