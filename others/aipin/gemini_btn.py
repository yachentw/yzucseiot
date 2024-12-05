import google.generativeai as genai
from gtts import gTTS
import subprocess
import tempfile
import cv2
import PIL.Image
import RPi.GPIO as GPIO
import time

# 設定 API 金鑰
genai.configure(api_key="...")

# GPIO 設定
BUTTON_PIN = 11  # 按鈕連接的 GPIO 腳位 (BOARD 編號)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 拍照功能
def capture_image(output_path):
    print("使用 OpenCV (USB 相機) 拍照...")
    cap = cv2.VideoCapture(0)  # 使用預設相機
    if not cap.isOpened():
        raise RuntimeError("無法打開相機")
    
    # 捕捉影像
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("無法捕獲影像")
    
    # 儲存影像
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"影像已儲存至 {output_path}")

# 檢查按鈕長按
def wait_for_long_press(pin, duration=3):
    start_time = None
    while True:
        if GPIO.input(pin) == GPIO.LOW:  # 按鈕被按下
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= duration:
                print("按鈕長按成功！")
                return True
        else:
            start_time = None  # 重置計時器
        time.sleep(0.1)

# 主程式
def main_loop():
    try:
        while True:
            print("等待按鈕觸發...")
            if wait_for_long_press(BUTTON_PIN):
                # Play the 'ding' sound using aplay
                subprocess.run(['aplay', "ding.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 拍照並儲存影像
                image_path = "captured_image.jpg"
                capture_image(image_path)

                # 載入影像
                photo = PIL.Image.open(image_path)

                # 初始化生成模型
                model = genai.GenerativeModel("gemini-1.5-flash")

                # 生成內容
                response = model.generate_content(["請簡短說明圖片上面有什麼東西", photo])
                generated_text = response.text

                # 輸出生成的文字
                print("生成的文字:", generated_text)

                # 將生成的文字轉換為語音並播放
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_path = temp_file.name
                    tts = gTTS(text=generated_text, lang="zh-tw")
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
    except KeyboardInterrupt:
        print("程式中止。")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()