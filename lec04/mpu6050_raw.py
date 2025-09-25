import time
from enum import Enum
import RPi.GPIO as GPIO
import struct
import smbus

class Accel(Enum):
    accel_2g = 0x00
    accel_4g = 0x01
    accel_8g = 0x02
    accel_16g = 0x03

class Gyro(Enum):
    gyro_250s = 0x00
    gyro_500s = 0x01
    gyro_1000s = 0x02
    gyro_2000s = 0x03

MPU6050_ADDRESS = 0x68  # MPU6050 I2C address
WHO_AM_I = 0x75
I_AM = 0x68
PWR_MGMT_1 = 0x6B
I2C_MST_CTRL = 0x24
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47
RA_CONFIG = 0x1A
RA_SMPLRT_DIV = 0x19
RA_GYRO_CONFIG = 0x1B
RA_ACCEL_CONFIG = 0x1C
RA_INT_PIN_CFG = 0x37
RA_INT_ENABLE = 0x38
RA_SIGNAL_PATH_RESET = 0x68
GYRO_XOFF_H = 0x13
GYRO_YOFF_H = 0x15
GYRO_ZOFF_H = 0x17
ACCEL_XOFF_H = 0x06
ACCEL_YOFF_H = 0x08
ACCEL_ZOFF_H = 0x0A

accelSensitivity = Accel.accel_2g
gyroSensitivity = Gyro.gyro_250s
# samplingRate = 0x00
lowPassFilter = 0x06

cali_offset = {'ax' : 0, 'ay': 0, 'az' : 0, 'gx' : 0, 'gy': 0, 'gz' : 0}

bus = None

def init():
    global bus
    bus = smbus.SMBus(1)
    configModule()

    
def testWhoAmI():
    ans = bus.read_byte_data(MPU6050_ADDRESS, WHO_AM_I)
    print(ans)
    if ans == I_AM:
        return True
    else:
        return False

def writeMPU6050(reg, val):
    bus.write_byte_data(MPU6050_ADDRESS, reg, val)

def writeRegBytes(reg, vals):
    print(vals)
    bus.write_i2c_block_data(MPU6050_ADDRESS, reg, vals)

def readMPU6050(reg, byteNum):
    return bus.read_i2c_block_data(MPU6050_ADDRESS, reg, byteNum)

def calibrate_accel(samples=100):
    print("calibrating accel")
    x_offset, y_offset, z_offset = 0, 0, 0
    for i in range(samples):
        axo, = struct.unpack('>h', bytes(readMPU6050(ACCEL_XOUT_H, 2)))
        ayo, = struct.unpack('>h', bytes(readMPU6050(ACCEL_YOUT_H, 2)))
        azo, = struct.unpack('>h', bytes(readMPU6050(ACCEL_ZOUT_H, 2)))
        x_offset += axo
        y_offset += ayo
        z_offset += (azo - 16384)
        time.sleep(0.01)
    x_offset /= samples
    y_offset /= samples
    z_offset /= samples
    return x_offset, y_offset, z_offset

def calibrate_gyro(samples=100):
    print("calibrating gyro")
    x_offset, y_offset, z_offset = 0, 0, 0
    for i in range(samples):
        gxo, = struct.unpack('>h', bytes(readMPU6050(GYRO_XOUT_H, 2)))
        gyo, = struct.unpack('>h', bytes(readMPU6050(GYRO_YOUT_H, 2)))
        gzo, = struct.unpack('>h', bytes(readMPU6050(GYRO_ZOUT_H, 2)))
        x_offset += gxo
        y_offset += gyo
        z_offset += gzo
        time.sleep(0.01)
    x_offset /= samples
    y_offset /= samples
    z_offset /= samples
    return x_offset, y_offset, z_offset
    
def configModule():
    writeMPU6050(PWR_MGMT_1, 0x80) # 1<<7 reset the whole module first
    time.sleep(0.05)
    writeMPU6050(PWR_MGMT_1, 0x03) # PLL with Z axis gyroscope reference
    time.sleep(0.05)
    # writeMPU6050(I2C_MST_CTRL, 0x07)
    # time.sleep(0.05)
    # writeMPU6050(RA_SMPLRT_DIV, samplingRate)
    # time.sleep(0.05)
    writeMPU6050(RA_CONFIG, lowPassFilter)
    time.sleep(0.05)
    writeMPU6050(RA_GYRO_CONFIG, gyroSensitivity.value << 3) # Gyro full scale setting
    time.sleep(0.05)
    writeMPU6050(RA_ACCEL_CONFIG, accelSensitivity.value << 3) # Accel full scale setting
    time.sleep(0.05)

    cali_offset['gx'], cali_offset['gy'], cali_offset['gz'] = calibrate_gyro()

    # writeMPU6050(RA_INT_PIN_CFG, 0x10) # 1<<4 interrupt status bits are cleared on any read operation
    # time.sleep(0.05)
    # writeMPU6050(RA_INT_ENABLE, 0x01) # interupt occurs when data is ready.
    # time.sleep(0.05)
    writeMPU6050(RA_SIGNAL_PATH_RESET, 0x07) # reset gyro and accel sensor
    time.sleep(0.05)



def readMPU6050All():
    return bus.read_i2c_block_data(MPU6050_ADDRESS, ACCEL_XOUT_H, 14)


def close():
    writeMPU6050(PWR_MGMT_1, 0x80) # 1<<7 reset the whole module first


if __name__ == "__main__":
    init()
    # pass the whoami test for defected modules
    # if testWhoAmI:
    try:
        while True:
            accel = {'x': 0, 'y': 0, 'z': 0}
            gyro = {'x': 0, 'y': 0, 'z': 0}

            data = readMPU6050All()        
            
            # X-Axis
            # data0 = bus.read_byte_data(MPU6050_ADDRESS, ACCEL_XOUT_H)
            # data1 = bus.read_byte_data(MPU6050_ADDRESS, ACCEL_XOUT_H + 1)
            xAccl = struct.unpack('>h', bytes([data[0], data[1]]))[0]  # Big-endian (MSB first)
            accel['x'] = xAccl / 16384.0  # scale factor is 16384
            # Y-Axis
            yAccl = struct.unpack('>h', bytes([data[2], data[3]]))[0]
            accel['y'] = yAccl / 16384.0
            # Z-Axis
            zAccl = struct.unpack('>h', bytes([data[4], data[5]]))[0]
            accel['z'] = zAccl / 16384.0

            # gyro X-Axis
            xGyro = struct.unpack('>h', bytes([data[8], data[9]]))[0]  # Big-endian (MSB first)
            gyro['x'] = (xGyro - cali_offset['gx']) / 131.0
            # gyro Y-Axis
            yGryo = struct.unpack('>h', bytes([data[10], data[11]]))[0]
            gyro['y'] = (yGryo - cali_offset['gy']) / 131.0
            # gyro Z-Axis
            zGyro = struct.unpack('>h', bytes([data[12], data[13]]))[0]
            gyro['z'] = (zGyro - cali_offset['gz']) / 131.0

            # Output accelerometer data to the screen
            print("Ax Ay Az Gx Gy Gz: %.3f %.3f %.3f %.3f %.3f %.3f" % (accel['x'], accel['y'], accel['z'], gyro['x'], gyro['y'], gyro['z']))

            # Sleep for 100ms
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupted")
        
    # else:
    #     print('WHOAMI fail!')