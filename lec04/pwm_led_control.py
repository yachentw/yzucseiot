import RPi.GPIO as GPIO
# import time

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

pwm = GPIO.PWM(LED_PIN, 100)
pwm.start(0)
try:
    while True:
        brightness = input("Set brightness (0 ~ 100): ")
        if not brightness.isdigit() or int(brightness) > 100 or int(brightness) < 0:
            print("Please enter an integer between 0 ~ 100.")
            continue
        pwm.ChangeDutyCycle(int(brightness))
except KeyboardInterrupt:
    print("\nKeyboardInterrupt")
finally:
    pwm.stop()
    # for rpi-lgpio
    # GPIO.cleanup()

