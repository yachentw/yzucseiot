import time
import serial

ser = serial.Serial('/dev/ttyAMA5', baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
try:
    ser.write(b'Hello World\r\n')
    ser.write(b'Serial Communication Using Raspberry Pi\r\n')
    while True:
        data = ser.readline()
        print(data.decode("utf-8").strip())
        ser.write(data)
        ser.flushInput()
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    ser.close()
