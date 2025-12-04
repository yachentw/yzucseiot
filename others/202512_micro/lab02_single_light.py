import cv2
import threading
import time
import numpy as np
import tflite_runtime.interpreter as tflite
from gpiozero import LED  # 樹莓派 GPIO 控制庫
from tools import CustomVideoCapture, preprocess, parse_output

class SmartLightController:
    """
    基於 Teachable Machine 的 AI 智慧燈控系統
    對應投影片邏輯：
    1. 剪刀 -> 恆亮 (Steady)
    2. 石頭 -> 閃爍 (Blink)
    3. 背景/布 -> 熄滅 (Off)
    """

    # 配置常數
    MODEL_PATH = "model.tflite"      # 請確認您的模型檔名
    LABELS_PATH = "labels.txt"       # 請確認您的標籤檔名
    LED_PIN = 17                     # 設定 LED 接在 GPIO 17
    BLINK_INTERVAL = 0.5             # 閃爍頻率 (秒)

    def __init__(self):
        """初始化偵測器與硬體"""
        self.interpreter = None
        self.labels = []
        self.vid = None
        
        # --- 硬體與狀態控制 ---
        # 狀態變數: "OFF", "STEADY", "BLINK"
        self.current_light_mode = "OFF"
        self.running = True
        
        # 初始化 LED (使用 gpiozero)
        try:
            self.led = LED(self.LED_PIN)
            print(f"✓ GPIO {self.LED_PIN} 初始化成功")
        except Exception as e:
            print(f"⚠ GPIO 初始化失敗 (若在非樹莓派環境可忽略): {e}")
            self.led = None

        # 啟動獨立的 LED 控制執行緒
        self.led_thread = threading.Thread(target=self._led_control_loop, daemon=True)
        self.led_thread.start()

        # 載入模型
        self._load_model()
        self._load_labels()

    def _load_model(self):
        """載入 TensorFlow Lite 模型 (參考 main.py)"""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✓ 模型載入成功: {self.MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"模型載入失敗: {e}")

    def _load_labels(self):
        """載入標籤檔 (參考 main.py)"""
        try:
            with open(self.LABELS_PATH, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✓ 標籤載入成功: {self.labels}")
        except Exception as e:
            raise RuntimeError(f"標籤載入失敗: {e}")

    def _led_control_loop(self):
        """
        [獨立執行緒] 負責控制 LED 亮滅
        對應投影片中 '來自【實作1】的超能力！ while True: ...' 部分
        """
        while self.running:
            if self.led is None:
                time.sleep(1)
                continue

            if self.current_light_mode == "STEADY":
                # 剪刀 -> 恆亮
                self.led.on()
                time.sleep(0.1)  # 短暫休眠避免 CPU 滿載

            elif self.current_light_mode == "BLINK":
                # 石頭 -> 閃爍
                self.led.on()
                time.sleep(self.BLINK_INTERVAL)
                self.led.off()
                time.sleep(self.BLINK_INTERVAL)

            else: # OFF
                # 背景 -> 熄滅
                self.led.off()
                time.sleep(0.1)

    def process_frame(self, frame):
        """影像推論處理 (參考 main.py process_frame)"""
        # 1. 前處理
        data = preprocess(frame, resize=(224, 224), norm=True)

        # 2. 推論
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # 3. 解析結果
        # parse_output 通常回傳 (id, name, probability)
        trg_id, trg_class, trg_prob = parse_output(prediction, self.labels)
        return trg_class, trg_prob

    def update_light_mode(self, detected_class):
        """
        根據辨識結果更新燈號模式
        對應投影片中的決策邏輯
        """
        # 注意：這裡的字串 "Scissors", "Rock" 需對應您的 labels.txt 內容
        # 建議 labels.txt 內容為：
        # 0 Scissors
        # 1 Rock
        # 2 Background
        
        if "Scissors" in detected_class or "剪刀" in detected_class:
            self.current_light_mode = "STEADY"
        elif "Rock" in detected_class or "石頭" in detected_class:
            self.current_light_mode = "BLINK"
        else:
            # 包含 "Background", "Paper", "布" 等
            self.current_light_mode = "OFF"

    def run(self):
        """主程式迴圈"""
        try:
            self.vid = CustomVideoCapture()
            self.vid.set_title('AI Light Control')
            self.vid.start_stream()
            print("✓ 系統啟動，按 Esc 離開")

            while not self.vid.isStop:
                ret, frame = self.vid.get_current_frame()
                if not ret:
                    continue

                # 進行辨識
                detected_class, prob = self.process_frame(frame)
                
                # 更新 LED 狀態
                self.update_light_mode(detected_class)

                # 顯示資訊
                status_text = f"Mode: {self.current_light_mode}"
                self.vid.info = f'{detected_class} ({prob:.2f}) | {status_text}'

        except KeyboardInterrupt:
            print("程式中斷")
        except Exception as e:
            print(f"執行錯誤: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理資源"""
        self.running = False  # 停止 LED 執行緒
        if self.led:
            self.led.off()
            self.led.close()
        if self.vid:
            self.vid.stop_stream()
        print("✓ 程式與資源已釋放")

if __name__ == "__main__":
    app = SmartLightController()
    app.run()