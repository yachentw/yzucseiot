import RPi.GPIO as GPIO
import time

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)
pwm = GPIO.PWM(LED_PIN, 100)
pwm.start(0)

try:
    while True:
        for i in range(101):
            pwm.ChangeDutyCycle(i)
            time.sleep(0.05)
        pwm.ChangeDutyCycle(0)
        time.sleep(1)
except KeyboardInterrupt:
    pass
pwm.stop()
GPIO.cleanup()
