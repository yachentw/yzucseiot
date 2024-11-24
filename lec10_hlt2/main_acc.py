import io
import threading
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from queue import Queue
from tflite_runtime.interpreter import Interpreter
import mpu6050_raw
from PIL import Image

# Configurable parameters
sampling_interval = 0.01  # Sampling rate in seconds
window_size = 0.5  # Window size in seconds for each plot
slide_interval = 0.1  # Sliding interval in seconds for the plots
samples_num = int(window_size // sampling_interval)

actionState = []
currentState = 0 # relax -> move -> curl -> move -> relax
Curl_Count = 0
transCount = 0
missCount = 0
labels = ["Relax", "Move", "Curl"]
# actionState = ["Relax", "Move", "Curl", "Move", "Relax"]
actionState = ["Relax", "Curl", "Relax"]

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

def update_plot(fig, accel_lines, data_points, bg=None):
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

    if bg:
        fig.canvas.restore_region(bg)
    for line in accel_lines:
        ax.draw_artist(line)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

def sensor_reader(queue, stop_event):
    start_time = time.time()
    while True:
        if not stop_event.is_set():
            try:
                accel, _ = mpu6050_raw.getAccelGyro()  # Ignore gyroscope data
                timestamp = time.time()
                elapsed_time = timestamp - start_time
                queue.put({'time': elapsed_time, 'accel': accel})
                time.sleep(sampling_interval)
            except Exception as e:
                print(f"Error in sensor_reader: {e}")
                break

def preprocess(frame, resize=(224, 224), norm=True):
    frame_rgb = frame[:, :, :3]
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cropped_image = frame_rgb[:, 200:800]  # Crop region of interest
    input_format = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #  frame_resize = cv2.resize(cropped_image, resize)

    frame_norm = ((frame_rgb.astype(np.float32) / 127.5) -1) if norm else frame_rgb
    input_format[0] = frame_norm
    return input_format

if __name__ == "__main__":
    mpu6050_raw.init()
    data_queue = Queue()
    stop_event = threading.Event()

    interpreter = Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        stop_event.clear()
        sensor_thread = threading.Thread(target=sensor_reader, args=(data_queue, stop_event), daemon=True)
        sensor_thread.start()

        fig, ax, accel_lines = initialize_plot()
        plt.ion()

        recorded_data = []
        while True:
            while len(recorded_data) < samples_num:
                recorded_data.append(data_queue.get())

            if len(recorded_data) >= samples_num:
                # Pause the sensor thread and clear the queue
                stop_event.set()

                update_plot(fig, accel_lines, recorded_data[:samples_num])
                # Save plot and preprocess image
                # plt.savefig("temp_plot.png")
                # frame = cv2.imread("temp_plot.png")
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frame = np.array(Image.open(buf))
                buf.close()
                input_data = preprocess(frame)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                

                trg_class = labels[np.argmax(prediction)]
                print(f"Prediction: {trg_class} ({prediction})")
                print(trg_class, actionState[currentState])
                if trg_class == actionState[currentState]:
                    pass
                elif currentState < len(actionState) - 1 and trg_class == actionState[currentState + 1]:
                    transCount += 1
                    if transCount > 0:
                        currentState += 1
                        transCount = 0
                        if currentState == len(actionState) - 1:
                            currentState = 0
                            Curl_Count += 1
                # else:
                #     missCount += 1
                #     if missCount > 2:
                #         missCount = 0
                #         currentState = 0
                print("currentState:", currentState)
                print("curl count: ", Curl_Count)

                recorded_data = []
                with data_queue.mutex:
                    data_queue.queue.clear()
                # Restart the sensor thread
                stop_event.clear()

    except KeyboardInterrupt:
        print("\nExiting program.")
        stop_event.set()
        # sensor_thread.join()
        plt.ioff()
        plt.show()