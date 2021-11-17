import RPi.GPIO as GPIO
import socket

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

HOST = '0.0.0.0'
PORT = 8000
socks = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socks.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
socks.bind((HOST, PORT))
socks.listen(5)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')

try:
    while True:
        conn, addr = socks.accept()
        print('connected by ' + str(addr))
        while True:
            indata = conn.recv(1024)
            if len(indata) == 0: # connection closed
                conn.close()
                print('client closed connection.')
                break
            data = indata.decode("utf-8").strip()
            print('recv: %s' % data)
            if "1" in data:
                print("led on.")
                GPIO.output(LED_PIN, GPIO.HIGH)
            elif "0" in data:
                print("led off.")
                GPIO.output(LED_PIN, GPIO.LOW)
            # outdata = 'echo ' + indata.decode()
            # conn.send(outdata.encode())
except KeyboardInterrupt:
    print("Exception: KeyboardInterrupt")
finally:
    GPIO.cleanup()