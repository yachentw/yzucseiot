import subprocess
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import tempfile

# 初始化 Google Generative AI
API_KEY = "..."  # 請替換為您的實際 API Key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def translate_chinese_to_english(text):
    """
    將中文文本翻譯成英文

    Args:
        text: 要翻譯的中文文本

    Returns:
        翻譯後的英文文本
    """
    prompt = f"請將以下中文翻譯成英文，並請給我一個翻譯內容就好：{text}"
    response = model.generate_content(prompt)
    return response.text

def recognize_speech():
    """
    使用麥克風辨識語音並轉換成文字

    Returns:
        辨識出的文字 (str)
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請開始說話...")
        try:
            audio = recognizer.listen(source)
            print("正在辨識語音...")
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"辨識結果：{text}")
            return text
        except sr.UnknownValueError:
            print("無法辨識語音。")
            return None
        except sr.RequestError as e:
            print(f"語音辨識服務出錯：{e}")
            return None

def play_translation(input_text):
    """
    使用 gTTS 播放翻譯後的文字

    Args:
        text: 翻譯後的英文文本
    """
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
        tts = gTTS(text=input_text, lang="en-US")
        tts.save(temp_path)

        # 播放音訊檔案
        try:
            subprocess.run(
                ['vlc', '--play-and-exit', '--rate', '1.5', temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("VLC 未安裝或未找到 PATH 中的 VLC 可執行檔")

if __name__ == "__main__":
    # 語音輸入
    chinese_text = recognize_speech()
    if chinese_text:
        # 翻譯
        english_translation = translate_chinese_to_english(chinese_text)
        print(f"翻譯結果：{english_translation}")
        # 播放翻譯結果
        play_translation(english_translation)
    else:
        print("未能成功辨識語音或翻譯。")
