from bluezero import broadcaster
from bluezero import adapter
import time

def main():
    # 1. 取得藍牙裝置
    dongles = list(adapter.Adapter.available())
    if not dongles:
        print("Error：BT Device not found！")
        return
    adapter_address = dongles[0].address
    print(f"Using BT device: {adapter_address}")

    # 2. 建立 Beacon 廣播物件
    # 修正點：建構函式不放 local_name
    beacon = broadcaster.Beacon(adapter_address)

    # 3. 設定廣播屬性 (修正點：在建立物件後設定)
    beacon.local_name = 'RPi_Student_ID'  # 在這裡設定名稱
    
    # 加入廠商資料 (0xFFFF + 學號)
    # 請將 b'\x11\x22\x33\x44' 改為您的學號後四碼
    beacon.add_manufacturer_data(0xFFFF, b'\x11\x22\x33\x44')

    # 4. 開始廣播
    print(f"Advertising... Name: {beacon.local_name}")
    print("Ctrl+C to stop")
    
    try:
        beacon.start_beacon()
    except KeyboardInterrupt:
        beacon.stop_beacon()
        print("\nAdvertising stopped.")

if __name__ == '__main__':
    main()