import time
import requests

# Replace with your token
BLYNK_AUTH_TOKEN = '...'
VIRTUAL_PIN = 'V0'

# Blynk API URL
BLYNK_GET_URL = f'https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&{VIRTUAL_PIN}'

def get_virtual_pin_value():
    """Get the lastest value of a virtual pin"""
    while True:
        try:
            response = requests.get(BLYNK_GET_URL)
            if response.status_code == 200:
                value = response.text.strip()
                print(f"Virtual PIN {VIRTUAL_PIN}: {value}")
                return value
            else:
                print(f"Cannot access: {response.status_code}")
        except Exception as e:
            print(f"Exception: {e}")
        time.sleep(10)

if __name__ == '__main__':
    if BLYNK_AUTH_TOKEN == "...":
        print("Replace the token.")
    else:
        get_virtual_pin_value()
