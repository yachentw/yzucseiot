import time
import requests
import RPi.GPIO as GPIO
import sys

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

# Replace with your token
BLYNK_AUTH_TOKEN = '...'
VIRTUAL_PIN = 'V1'

# Blynk API URL
BLYNK_GET_URL = f'https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&{VIRTUAL_PIN}'

def led(cmd):
    if cmd == 1:
        print("led on.")
        GPIO.output(LED_PIN, GPIO.HIGH)
    elif cmd == 0:
        print("led off.")
        GPIO.output(LED_PIN, GPIO.LOW)


def get_virtual_pin_value():
    """Get the lastest value of a virtual pin"""
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

if __name__ == '__main__':
    if BLYNK_AUTH_TOKEN == "...":
        print("Replace the token.")
        GPIO.cleanup()
        sys.exit()
    while True:
        cmd = get_virtual_pin_value()
        led(int(cmd))
        time.sleep(1)
