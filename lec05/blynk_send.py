import requests
import time

# Replace with your token
BLYNK_AUTH_TOKEN = '...'
VIRTUAL_PIN = 'V0'

# Blynk API URL
BLYNK_URL = f'https://blynk.cloud/external/api/update?token={BLYNK_AUTH_TOKEN}&{VIRTUAL_PIN}='

def upload_data(value):
    try:
        url = BLYNK_URL + str(value)
        response = requests.get(url)

        if response.status_code == 200:
            print(f"Update: {value}")
        else:
            print(f"Update failure: {response.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

def main():
    """Upload a simulated temperature every 10 seconds"""
    while True:
        temperature = round(20 + 10 * (time.time() % 60) / 60, 2)
        upload_data(temperature)
        time.sleep(10)

if __name__ == '__main__':
    if BLYNK_AUTH_TOKEN == "...":
        print("Replace the token.")
    else:
        main()
