from flask import Flask, request
import RPi.GPIO as GPIO

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

app = Flask(__name__)

def led(cmd):
    if cmd == 1:
        print("led on.")
        GPIO.output(LED_PIN, GPIO.HIGH)
    elif cmd == 0:
        print("led off.")
        GPIO.output(LED_PIN, GPIO.LOW)
                
@app.route('/')
def index():
    return "Index Page"
@app.route('/cmd/<int:num>')
def cmd(num):
    led(num)    
    return "cmd: %d" % (num)

@app.route('/led', methods=['GET'])
def ledcmd():
    if request.method == 'GET':
        cmd = request.args.get('on')
        led(int(cmd))
        return "LED on: %s" % cmd

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("Exception: KeyboardInterrupt")
    finally:
        GPIO.cleanup()