import RPi.GPIO as GPIO
import time

def ButtonPressed(btn):
    print("Button pressed @", time.ctime())
    
GPIO.setmode(GPIO.BOARD)
BTN_PIN = 11
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(BTN_PIN, GPIO.FALLING, ButtonPressed, 200)
try:    
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exception: KeyboardInterrupt")
finally:
    GPIO.cleanup()

