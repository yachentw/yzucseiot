import google.generativeai as genai
from gtts import gTTS
import subprocess
import tempfile
import cv2
import PIL.Image

# 設定 API 金鑰
genai.configure(api_key="...")

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
