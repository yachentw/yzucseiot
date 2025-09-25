import RPi.GPIO as GPIO
import time

LED_PINS = [12, 11]
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PINS[0], GPIO.OUT)
GPIO.setup(LED_PINS[1], GPIO.OUT)
ledNum = len(LED_PINS)
try:
    counter = 1
    while True:
        for i in range(ledNum):
            if (counter >> i) & 0x00000001:
                GPIO.output(LED_PINS[i], GPIO.HIGH)
            else:
                GPIO.output(LED_PINS[i], GPIO.LOW)        
        counter = counter << 1
        if counter > ledNum:
            counter = 1
        time.sleep(1)
except KeyboardInterrupt:
    print("Exception: KeyboardInterrupt")
finally:
    GPIO.cleanup()
