# genai.configure(api_key="AIzaSyAhQNfZeNH0jPbBctginKmWCQDsipZNXbQ")

import google.generativeai as genai
from gtts import gTTS
import PIL.Image
import subprocess
import tempfile

# 設定 API 金鑰
genai.configure(api_key="...")

# 載入圖片
organ = PIL.Image.open("organ.jpg")

# 初始化模型
model = genai.GenerativeModel("gemini-1.5-flash")

# 生成內容
response = model.generate_content(["請簡短告訴我關於這個樂器", organ])
generated_text = response.text

# 輸出生成的文字
print("生成的文字:", generated_text)

# Generate temporary file with delete=True
with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
    temp_path = temp_file.name
    tts = gTTS(text=generated_text, lang="zh-tw")
    tts.save(temp_path)

    # Play the MP3 file
    try:
        subprocess.run(
            ['vlc', '--play-and-exit', '--rate', '1.5', temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("VLC is not installed or not found in the PATH.")