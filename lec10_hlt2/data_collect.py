import time
import matplotlib.pyplot as plt
import mpu6050_raw
import threading
from queue import Queue

# Configurable parameters
sampling_interval = 0.01  # Sampling rate in seconds
window_size = 0.5  # Window size in seconds for each plot
slide_interval = 0.1  # Sliding interval in seconds for the plots
record_duration = 10

def initialize_plot():
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    accel_lines = [
        ax.plot([], [], label='Accel X')[0],
        ax.plot([], [], label='Accel Y')[0],
        ax.plot([], [], label='Accel Z')[0]
    ]

    # Configure accelerometer plot
    ax.xaxis.set_visible(False)
    ax.set_ylim(-1.25, 1.25)
    # ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))

    plt.tight_layout()
    return fig, ax, accel_lines

def update_plot(fig, accel_lines, data_points, directory, plot_counter):
    timestamps = [point['time'] for point in data_points]
    accel_data = [
        [point['accel']['x'] for point in data_points],
        [point['accel']['y'] for point in data_points],
        [point['accel']['z'] for point in data_points]
    ]

    for line, data in zip(accel_lines, accel_data):
        line.set_data(timestamps, data)

    ax = fig.axes[0]
    ax.set_xlim(timestamps[0], timestamps[-1])

    fig.savefig(f"{directory}/{plot_counter}.png")
    print(f"Saved {directory}/{plot_counter}.png")

def sensor_reader(queue, stop_event, seconds):
    start_time = time.time()
    qcount = 0
    collectsz = seconds // sampling_interval
    while not stop_event.is_set():
        accel, _ = mpu6050_raw.getAccelGyro()  # Only fetch accelerometer data
        timestamp = time.time()
        elapsed_time = timestamp - start_time
        queue.put({'time': elapsed_time, 'accel': accel})
        qcount += 1
        if qcount > collectsz:
            break
        time.sleep(sampling_interval)  # Collect data at the specified interval

import os

if __name__ == "__main__":
    mpu6050_raw.init()
    data_queue = Queue()
    stop_event = threading.Event()

    print("Enter a directory name to start recording data.")

    try:
        while True:
            user_input = input("Enter a directory name to record or 'q' to quit: ").strip()

            if user_input.lower() == 'q':
                print("Exiting program.")
                break

            elif user_input:
                plot_counter = 1
                directory = user_input
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print(f"Recording started. Saving to directory: {directory}")

                # Start sensor reading thread
                stop_event.clear()
                sensor_thread = threading.Thread(target=sensor_reader, args=(data_queue, stop_event, record_duration))
                sensor_thread.start()

                print("Recording sensor data for %d seconds..." % record_duration)
  
                # Collect data for 10 seconds
                sensor_thread.join()

                recorded_data = []
                start_time = time.time()
                print("q_size: ", data_queue.qsize())
                
                while not data_queue.empty():
                    recorded_data.append(data_queue.get())
                
                print("recorded: ", len(recorded_data))
                print("Sensor data recording complete.")

                # Stop sensor thread
                stop_event.set()
                sensor_thread.join()

                print("Generating plots...")

                # Initialize plot
                fig, axes, accel_lines = initialize_plot()

                # Generate plots using sliding window and specified parameters
                for i in range(0, len(recorded_data) - int(window_size / sampling_interval), int(slide_interval / sampling_interval)):
                    data_buffer = recorded_data[i:i + int(window_size / sampling_interval)]
                    update_plot(fig, accel_lines, data_buffer, directory, plot_counter)
                    plot_counter += 1

                print("All plots generated.")

            else:
                print("Invalid input. Enter a directory name to record or 'q' to quit.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupted. Exiting program.")
        stop_event.set()
        if 'sensor_thread' in locals():
            sensor_thread.join()
