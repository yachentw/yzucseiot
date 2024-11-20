import apa102
import time
LED_NUM = 3
leds = apa102.APA102(num_led=3)
colors = [[255,0,0],[0,255,0],[0,0,255]] # LED0: R, LED1: G, LED2: B
try:
    while True:
        for i in range(LED_NUM):
            leds.set_pixel(i, colors[i][0], colors[i][1], colors[i][2], 10)
        leds.show()
        time.sleep(1)
        leds.clear_strip()
        time.sleep(1)
except:
    pass
finally:
    leds.clear_strip()
    leds.cleanup()
