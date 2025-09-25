from mpu6050 import mpu6050
import time


sensor = mpu6050(0x68)

while True:
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()
    print(f"acceleration: X = {accel_data['x']:.2f} g, Y = {accel_data['y']:.2f} g, Z = {accel_data['z']:.2f} g")
    print(f"angular velocity: X = {gyro_data['x']:.2f} °/s, Y = {gyro_data['y']:.2f} °/s, Z = {gyro_data['z']:.2f} °/s")
    time.sleep(0.1)
