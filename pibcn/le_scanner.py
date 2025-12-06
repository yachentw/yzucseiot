import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning 10 seconds..")
    
    # === 對應原本的 scanner.scan(10.0) ===
    # return_adv=True 會同時回傳「裝置資訊」與「廣播封包(含RSSI/學號)」
    # 這會是一個 Dictionary: { MAC地址: (device, adv_data) }
    scanned_results = await BleakScanner.discover(timeout=10.0, return_adv=True)
    
    print(f"Scan finished, found {len(scanned_results)} devices.\n")

    # === 對應原本的 for dev in devices: ===
    for device_address, (device, adv_data) in scanned_results.items():
        
        # 檢查是否有廠商資料 (Manufacturer Data)
        if adv_data.manufacturer_data:
            # 遍歷這台裝置的所有廠商 ID
            for m_id, m_data in adv_data.manufacturer_data.items():
                
                # === 這裡是 Lab 的判斷邏輯 (找 0xFFFF) ===
                if m_id == 0xFFFF:
                    print("-" * 40)
                    print(f"[★] Found Beacon！")
                    
                    # 顯示名稱 (防呆處理)
                    name = device.name if device.name else "Unknown"
                    print(f"Device: {name}")
                    print(f"Addr  : {device.address}")
                    
                    # [重點] 從 adv_data 讀取 RSSI
                    print(f"RSSI  : {adv_data.rssi} dBm")
                    
                    # 顯示學號
                    print(f"Data  : 0x{m_data.hex()}")
                    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())