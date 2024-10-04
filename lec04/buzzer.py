import RPi.GPIO as GPIO
import time

BUZZ_PIN = 16
pitches = [262, 294, 330, 349, 392, 440, 493, 523]
# pitches = [262, 294, 330, 349, 392, 440, 493, 523, 587, 659, 698, 784, 880, 932, 988]
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUZZ_PIN, GPIO.OUT)

pwm = GPIO.PWM(BUZZ_PIN, pitches[0])
pwm.start(0)

def play(pitch, intv):
    pwm.ChangeFrequency(pitch)
    time.sleep(intv)
try:
    while True:
        pwm.ChangeDutyCycle(50)
        for pitch in pitches:
            play(pitch, 1)
        pwm.ChangeDutyCycle(0)
        time.sleep(5)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt")
finally:
    pwm.stop()
    # for rpi-lgpio
    # GPIO.cleanup()


