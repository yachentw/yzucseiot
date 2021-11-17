'''
This Example sends harcoded data to Ubidots using the requests
library.

Please install the library using pip install requests

Made by Jose García @https://github.com/jotathebest/
'''

import requests
import random
import time

'''
global variables
'''

ENDPOINT = "things.ubidots.com"
DEVICE_LABEL = "weather-station"
VARIABLE_LABEL = "temperature"
TOKEN = "..." # replace with your TOKEN
DELAY = 1  # Delay in seconds


def get_var(url=ENDPOINT, device=DEVICE_LABEL, variable=VARIABLE_LABEL,
            token=TOKEN):
    try:
        url = "http://{}/api/v1.6/devices/{}/{}/?page_size=2".format(url,
                                                        device,
                                                        variable)

        headers = {"X-Auth-Token": token, "Content-Type": "application/json"}

        attempts = 0
        status_code = 400

        while status_code >= 400 and attempts < 5:
            print("[INFO] Retrieving data, attempt number: {}".format(attempts))
            req = requests.get(url=url, headers=headers)
            status_code = req.status_code
            attempts += 1
            time.sleep(1)

        print("[INFO] Results:")
        print(req.text)
    except Exception as e:
        print("[ERROR] Error posting, details: {}".format(e))


if __name__ == "__main__":
    while True:
        get_var()
        time.sleep(DELAY)