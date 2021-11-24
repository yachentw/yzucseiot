
import requests
import random
import time
import base64
from camera_pi import Camera

'''
global variables
'''

ENDPOINT = "things.ubidots.com"
DEVICE_LABEL = "camera_rpi"
VARIABLE_LABEL = "image"
TOKEN = "..."
DELAY = 15  # Delay in seconds


def post_var(payload, url=ENDPOINT, device=DEVICE_LABEL, token=TOKEN):
    try:
        url = "http://{}/api/v1.6/devices/{}".format(url, device)
        headers = {"X-Auth-Token": token, "Content-Type": "application/json"}

        attempts = 0
        status_code = 400

        while status_code >= 400 and attempts < 5:
            print("[INFO] Sending data, attempt number: {}".format(attempts))
            req = requests.post(url=url, headers=headers,
                                json=payload)
            status_code = req.status_code
            attempts += 1
            time.sleep(1)

        print("[INFO] Results:")
        print(req.text)
    except Exception as e:
        print("[ERROR] Error posting, details: {}".format(e))

def capture(camera):
    img = camera.get_frame_b64()
    payload = {VARIABLE_LABEL: {"value" : len(img), "context" : {"image" : img}}}
    # print(payload)
    # Sends data
    post_var(payload)


if __name__ == "__main__":
    camera = Camera()
    while True:
        capture(camera)
        time.sleep(DELAY)
