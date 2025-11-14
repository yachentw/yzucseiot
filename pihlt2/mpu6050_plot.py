import smbus
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpu6050_raw

# Initialize MPU6050
mpu6050_raw.init()

# Configurable x-axis range
display_range = 5  # Time range in seconds

# Initialize plot
fig, (ax_accel, ax_gyro) = plt.subplots(2, 1, figsize=(6.4, 4.8))

# Data storage
accel_x_data, accel_y_data, accel_z_data = [], [], []
gyro_x_data, gyro_y_data, gyro_z_data = [], [], []
time_data = []

# Accelerometer plot setup
ax_accel.set_title("Accelerometer Data")
ax_accel.set_xlabel("Time (s)")
ax_accel.set_ylabel("Acceleration (g)")
ax_accel.set_ylim(-2, 2)
line_accel_x, = ax_accel.plot([], [], label="Accel X", color="blue", linewidth=2)
line_accel_y, = ax_accel.plot([], [], label="Accel Y", color="orange", linewidth=2)
line_accel_z, = ax_accel.plot([], [], label="Accel Z", color="green", linewidth=2)
ax_accel.legend()

# Gyroscope plot setup
ax_gyro.set_title("Gyroscope Data")
ax_gyro.set_xlabel("Time (s)")
ax_gyro.set_ylabel("Angular Velocity (°/s)")
ax_gyro.set_ylim(-200, 200)
line_gyro_x, = ax_gyro.plot([], [], label="Gyro X", color="blue", linewidth=2)
line_gyro_y, = ax_gyro.plot([], [], label="Gyro Y", color="orange", linewidth=2)
line_gyro_z, = ax_gyro.plot([], [], label="Gyro Z", color="green", linewidth=2)
ax_gyro.legend()

start_time = time.time()

# Update function for animation
def update(frame):
    global accel_x_data, accel_y_data, accel_z_data
    global gyro_x_data, gyro_y_data, gyro_z_data, time_data

    accel, gyro = mpu6050_raw.getAccelGyro()
    accel_x = accel['x']
    accel_y = accel['y']
    accel_z = accel['z']
    gyro_x = gyro['x']
    gyro_y = gyro['y']
    gyro_z = gyro['z']

    current_time = time.time() - start_time

    time_data.append(current_time)
    accel_x_data.append(accel_x)
    accel_y_data.append(accel_y)
    accel_z_data.append(accel_z)
    gyro_x_data.append(gyro_x)
    gyro_y_data.append(gyro_y)
    gyro_z_data.append(gyro_z)

    # Keep only the last data within display_range
    if current_time > display_range:
        time_data = time_data[-int(display_range * 10):]
        accel_x_data = accel_x_data[-len(time_data):]
        accel_y_data = accel_y_data[-len(time_data):]
        accel_z_data = accel_z_data[-len(time_data):]
        gyro_x_data = gyro_x_data[-len(time_data):]
        gyro_y_data = gyro_y_data[-len(time_data):]
        gyro_z_data = gyro_z_data[-len(time_data):]

    # Update accelerometer plot
    line_accel_x.set_data(time_data, accel_x_data)
    line_accel_y.set_data(time_data, accel_y_data)
    line_accel_z.set_data(time_data, accel_z_data)
    ax_accel.set_xlim(max(0, current_time - display_range), current_time)

    # Update gyroscope plot
    line_gyro_x.set_data(time_data, gyro_x_data)
    line_gyro_y.set_data(time_data, gyro_y_data)
    line_gyro_z.set_data(time_data, gyro_z_data)
    ax_gyro.set_xlim(max(0, current_time - display_range), current_time)

    return line_accel_x, line_accel_y, line_accel_z, line_gyro_x, line_gyro_y, line_gyro_z

ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

import signal
import sys

def signal_handler(sig, frame):
    print("\nKeyboardInterrupt: Exiting program.")
    sys.exit(0)

def on_key(event):
    if event.key == 'escape':
        print("ESC pressed: Exiting program.")
        plt.close(fig)
        sys.exit(0)

fig.canvas.mpl_connect('key_press_event', on_key)

signal.signal(signal.SIGINT, signal_handler)

plt.tight_layout()
plt.show()
