import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
BTN_PIN = 11
WAIT_TIME = 0.2
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
previousStatus = None
previousTime = time.time()
currentTime  = None

try:
    while True:
        input = GPIO.input(BTN_PIN)
        currentTime = time.time()
        if input == GPIO.LOW and previousStatus == GPIO.HIGH and (currentTime - previousTime) > WAIT_TIME:
            previousTime = currentTime
            print("Button pressed @", time.ctime())
        previousStatus = input

except KeyboardInterrupt:
    print("Exception: KeyboardInterrupt")

finally:
    GPIO.cleanup()