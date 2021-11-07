import pigpio
import time
PWM_PIN = 18 # BCM Number
pi = pigpio.pi()
pi.set_PWM_frequency(PWM_PIN, 100)
pi.set_PWM_range(PWM_PIN, 255)
pi.set_PWM_dutycycle(PWM_PIN, 0)
try:
    while True:
        for i in range(101):
            pi.set_PWM_dutycycle(PWM_PIN, 255*i/100)
            time.sleep(0.05)
except KeyboardInterrupt:
    pass
pi.set_PWM_dutycycle(PWM_PIN, 0)
pi.stop()
