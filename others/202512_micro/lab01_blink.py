import time
import RPi.GPIO as GPIO

# --- 設定 GPIO ---
# 使用 BCM 編號模式 (對應 GPIO 17)
GPIO.setmode(GPIO.BCM)
# 設定 GPIO 17 為輸出模式 (用來控制 LED)
GPIO.setup(17, GPIO.OUT)

print("程式啟動：LED 開始閃爍 (按 Ctrl+C 結束)")

try:
    # --- 核心觀念：閃爍的無限循環 ---
    while True:
        # 1. 亮
        GPIO.output(17, GPIO.HIGH)  # Turn LED on
        # 2. 等待 (這就是投影片提到的修改點)
        time.sleep(1)               # Wait for 1 second
        
        # 3. 滅
        GPIO.output(17, GPIO.LOW)   # Turn LED off
        # 4. 等待
        time.sleep(1)               # Wait for 1 second

except KeyboardInterrupt:
    # 當按下 Ctrl+C 時，清理 GPIO 設定並安全退出
    print("\n程式結束，清理 GPIO")
    GPIO.cleanup()