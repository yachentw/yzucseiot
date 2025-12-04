import os
import speech_recognition as sr
import google.generativeai as genai
from gTTS import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

# ================= 設定區 =================
# 請將這裡換成您的 Gemini API Key
GOOGLE_API_KEY = "您的_GEMINI_API_KEY"

# 設定 Gemini 模型 (建議使用 flash 版本，回應速度較快)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# 初始化語音辨識器
recognizer = sr.Recognizer()
# ==========================================

def speak(text):
    """
    將文字轉為語音並播放 (Text-to-Speech)
    """
    print(f"Gemini 正在說話: {text}")
    try:
        # 使用 gTTS 將文字轉為語音 (lang='zh-TW' 設定為台灣中文)
        tts = gTTS(text=text, lang='zh-TW')
        
        # 將語音存入記憶體 (不存成實體檔案以加快速度)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # 使用 pydub 播放
        song = AudioSegment.from_file(fp, format="mp3")
        play(song)
    except Exception as e:
        print(f"TTS Error: {e}")

def listen_to_voice():
    """
    聆聽麥克風輸入並轉為文字 (Speech-to-Text)
    """
    with sr.Microphone() as source:
        print("\n正在聆聽中... (請說話)")
        # 自動調整環境噪音基準
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            # 開始錄音，直到偵測到停頓
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("正在辨識...")
            
            # 使用 Google 語音辨識 (免費 API，需連網)
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"您說: {text}")
            return text
        except sr.WaitTimeoutError:
            print("沒有偵測到聲音。")
            return None
        except sr.UnknownValueError:
            print("聽不懂您說什麼。")
            return None
        except sr.RequestError as e:
            print(f"無法連線到語音辨識服務; {e}")
            return None

def chat_with_gemini(user_input):
    """
    發送文字給 Gemini 並取得回應
    """
    if not user_input:
        return None
    
    print("Gemini 思考中...")
    try:
        # 為了讓語音回應更自然，可以加一點 prompt 限制
        prompt = f"請用繁體中文回答，盡量口語化且簡短（適合語音朗讀）。使用者說：{user_input}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "抱歉，我現在無法連線到大腦。"

# ================= 主程式迴圈 =================
if __name__ == "__main__":
    print("=== Gemini 語音助理啟動 ===")
    print("按 Ctrl+C 結束程式")
    
    speak("你好，我是 Gemini 語音助理，請問有什麼我可以幫你的？")

    try:
        while True:
            # 1. 聽
            user_text = listen_to_voice()
            
            if user_text:
                # 簡單的結束指令
                if "再見" in user_text or "關機" in user_text:
                    speak("好的，再見。")
                    break

                # 2. 想
                ai_response = chat_with_gemini(user_text)
                
                # 3. 說
                if ai_response:
                    speak(ai_response)
            
    except KeyboardInterrupt:
        print("\n程式已結束")