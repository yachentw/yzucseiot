#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webcam Object Detection with Motor Control (TFLite) - Optimized Version
- Ready for EfficientDet-Lite0/1/2 (TF Hub) and SSD-MobileNet (V1/V2/V3) TFLite models
- Works on Raspberry Pi (Pi 4/5) with tflite-runtime or full TensorFlow
- Robust output mapping (auto-detects boxes/classes/scores/count regardless of name/order)
- Threaded VideoStream for better FPS
- Motor control for object tracking (following detected persons)

Optimizations:
- Robust detection output mapping
- Improved FPS calculation
- Better error handling and resource cleanup
- Threaded motor control to avoid blocking
"""
import os
import sys
import time
import argparse
import importlib.util
from threading import Thread

import cv2
import numpy as np

# Import motor control module
try:
    import pwm_motor as motor
    MOTOR_AVAILABLE = True
except ImportError:
    print("[WARN] pwm_motor module not found. Motor control disabled.")
    MOTOR_AVAILABLE = False

# ------------------------------
# Motor control thread
# ------------------------------
class MotorThread(Thread):
    """Thread to execute motor actions without blocking main loop"""
    def __init__(self, action):
        super().__init__(daemon=True)
        self.action = action

    def run(self):
        if not MOTOR_AVAILABLE:
            return
        
        try:
            if self.action == "turnLeft":
                motor.turnLeft()
            elif self.action == "turnRight":
                motor.turnRight()
            elif self.action == "forward":
                motor.forward()
            elif self.action == "backward":
                motor.backward()
            elif self.action == "stop":
                motor.stop()
        except Exception as e:
            print(f"[ERROR] Motor control failed: {e}")

# ------------------------------
# Video stream helper (threaded)
# ------------------------------
class VideoStream:
    def __init__(self, src=0, resolution=(640, 480), framerate=30, fourcc='MJPG'):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # Try to set format for better USB cams throughput
        if fourcc:
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# ------------------------------
# TFLite loader (tflite-runtime or TF)
# ------------------------------

def load_tflite_interpreter(model_path, use_tpu=False):
    """Load TFLite interpreter from tflite_runtime if available; otherwise from tensorflow."""
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        load_delegate = None
        if use_tpu:
            from tflite_runtime.interpreter import load_delegate as _ld
            load_delegate = _ld
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        load_delegate = None
        if use_tpu:
            from tensorflow.lite.python.interpreter import load_delegate as _ld
            load_delegate = _ld

    delegates = []
    # EdgeTPU delegate
    if use_tpu and load_delegate is not None:
        # common library names; on some systems it's libedgetpu.so.1
        for libname in ('libedgetpu.so.1.0', 'libedgetpu.so.1', 'libedgetpu.so'):
            try:
                delegates.append(load_delegate(libname))
                print(f"[INFO] Loaded EdgeTPU delegate: {libname}")
                break
            except Exception:
                continue
        if not delegates:
            print("[WARN] EdgeTPU delegate requested but not found")

    # Create interpreter with delegates if available
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    except TypeError:
        interpreter = Interpreter(model_path=model_path)

    return interpreter

# ------------------------------
# Output mapping (robust to names/order)
# ------------------------------

def map_detection_outputs(output_details):
    """Map output tensors to boxes, classes, scores, and count indices."""
    names = [d['name'] for d in output_details]
    n = len(output_details)

    def find_idx(keywords):
        for i, nm in enumerate(names):
            for k in keywords:
                if k in nm:
                    return i
        return None

    # First try by name contains
    boxes_idx   = find_idx(["detection_boxes", "boxes", "location"])
    classes_idx = find_idx(["detection_classes", "classes", "label"])
    scores_idx  = find_idx(["detection_scores", "scores"])
    count_idx   = find_idx(["num_detections", "count"])  # optional

    if (boxes_idx is not None) and (classes_idx is not None) and (scores_idx is not None):
        return boxes_idx, classes_idx, scores_idx, count_idx

    # Fallback by common orders
    if n == 3:  # [boxes, classes, scores]
        return 0, 1, 2, None
    if n == 4:  # [boxes, classes, scores, count] (TF Hub EfficientDet-Lite)
        return 0, 1, 2, 3

    raise RuntimeError(f"Unexpected number of outputs: {n}, names={names}")

# ------------------------------
# Drawing helpers with tracking
# ------------------------------

def draw_detections_with_tracking(frame, boxes, classes, scores, labels, 
                                  target_class="person", thr=0.5, 
                                  enable_motor=True):
    """
    Draw bounding boxes and labels on frame.
    If target_class is detected, calculate center and control motor.
    Returns True if target was found and motor was controlled.
    """
    imH, imW = frame.shape[:2]
    target_found = False
    
    for i in range(len(scores)):
        s = float(scores[i])
        if s < thr or s > 1.0:
            continue
        
        cls = int(classes[i]) if classes is not None else -1
        name = labels[cls] if 0 <= cls < len(labels) else str(cls)
        
        # Boxes are usually [ymin, xmin, ymax, xmax] normalized to [0,1]
        ymin = int(max(1,      boxes[i][0] * imH))
        xmin = int(max(1,      boxes[i][1] * imW))
        ymax = int(min(imH-1,  boxes[i][2] * imH))
        xmax = int(min(imW-1,  boxes[i][3] * imW))
        
        # Check if this is our target object
        is_target = (name.lower() == target_class.lower())
        
        # Draw rectangle (green for target, blue for others)
        color = (10, 255, 0) if is_target else (255, 100, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label
        label = f"{name}: {int(s*100)}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(ymin, th + 10)
        cv2.rectangle(frame, (xmin, y_text - th - 10), 
                     (xmin + tw, y_text + bl - 10), 
                     (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xmin, y_text - 7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # If this is our target and we haven't controlled motor yet
        if is_target and not target_found and enable_motor:
            # Calculate center point
            cx = int((xmin + xmax) // 2)
            cy = int((ymin + ymax) // 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
            
            # Draw vertical reference lines (35% and 65% of width)
            left_line = int(0.35 * imW)
            right_line = int(0.65 * imW)
            cv2.line(frame, (left_line, 0), (left_line, imH), (0, 255, 255), 1)
            cv2.line(frame, (right_line, 0), (right_line, imH), (0, 255, 255), 1)
            
            # Control motor based on position
            if cx < left_line:
                MotorThread("turnLeft").start()
                cv2.putText(frame, "TURN LEFT", (imW//2 - 100, imH - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif cx > right_line:
                MotorThread("turnRight").start()
                cv2.putText(frame, "TURN RIGHT", (imW//2 - 100, imH - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Target is centered, could move forward
                cv2.putText(frame, "CENTERED", (imW//2 - 100, imH - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            target_found = True
            break  # Only track the first detected target
    
    return target_found

def load_labels(labels_path):
    """Load label file, handling common formats."""
    if not os.path.exists(labels_path):
        print(f"[WARN] label file not found: {labels_path}. Using generic labels.")
        return [str(i) for i in range(1000)]
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [l.strip() for l in f if l.strip()]
    
    # Remove '???' placeholder if present (common in some label files)
    if labels and labels[0] == '???':
        labels = labels[1:]
    
    return labels

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modeldir', default='TFLite_model', help='Folder that contains the .tflite file and labels')
    ap.add_argument('--graph', default='eflite0.tflite', help='TFLite model filename')
    ap.add_argument('--labels', default='labelmap.txt', help='Label map filename (one class per line)')
    ap.add_argument('--threshold', type=float, default=0.5, help='Minimum confidence threshold')
    ap.add_argument('--resolution', default='640x480', help='Webcam resolution WxH')
    ap.add_argument('--edgetpu', action='store_true', help='Use Coral EdgeTPU delegate')
    ap.add_argument('--camera', type=int, default=0, help='Video device index (default 0)')
    ap.add_argument('--target', default='person', help='Target object class to track (default: person)')
    ap.add_argument('--no-motor', action='store_true', help='Disable motor control (for testing)')
    ap.add_argument('--force_input', default='', help='Override model input size as WxH (e.g., 320x320); empty to auto')
    args = ap.parse_args()

    # Setup paths
    model_path = os.path.join(os.getcwd(), args.modeldir, args.graph)
    labels_path = os.path.join(os.getcwd(), args.modeldir, args.labels)

    # Validate model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    # Load labels
    labels = load_labels(labels_path)

    # Check if target class exists in labels
    target_class = args.target
    if target_class.lower() not in [l.lower() for l in labels]:
        print(f"[WARN] Target class '{target_class}' not found in labels. Available labels: {labels[:10]}...")

    # Load interpreter
    print(f"[INFO] Loading model: {model_path}")
    interpreter = load_tflite_interpreter(model_path, use_tpu=args.edgetpu)

    # Initial allocation
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape and dtype
    in_h = int(input_details[0]['shape'][1])
    in_w = int(input_details[0]['shape'][2])
    in_dtype = input_details[0]['dtype']

    print(f'[INFO] Model input: {in_w}x{in_h}, dtype: {in_dtype}')
    print(f'[INFO] Model outputs: {len(output_details)} - {[d["name"] for d in output_details]}')

    # Handle force_input if specified
    if args.force_input:
        try:
            fw, fh = args.force_input.lower().split('x')
            fw, fh = int(fw), int(fh)
            print(f"[INFO] Resizing model input to {fw}x{fh}")
            interpreter.resize_tensor_input(input_details[0]['index'], [1, fh, fw, 3])
            interpreter.allocate_tensors()
            
            # Refresh details after resize
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            in_h = int(input_details[0]['shape'][1])
            in_w = int(input_details[0]['shape'][2])
            in_dtype = input_details[0]['dtype']
            print(f"[INFO] Model input resized to: {in_w}x{in_h}")
        except Exception as e:
            print(f"[WARN] --force_input ignored due to error: {e}")

    # Map outputs once after all potential resizing
    boxes_idx, classes_idx, scores_idx, count_idx = map_detection_outputs(output_details)
    print(f"[INFO] Output mapping - boxes:{boxes_idx}, classes:{classes_idx}, scores:{scores_idx}, count:{count_idx}")

    # Determine normalization parameters
    floating_model = (in_dtype == np.float32)
    input_mean, input_std = 127.5, 127.5

    # Prepare camera
    resW, resH = args.resolution.lower().split('x')
    imW, imH = int(resW), int(resH)
    print(f"[INFO] Starting camera {args.camera} at {imW}x{imH}")
    print(f"[INFO] Target object: {target_class}")
    print(f"[INFO] Motor control: {'DISABLED' if args.no_motor else 'ENABLED'}")
    
    vs = VideoStream(src=args.camera, resolution=(imW, imH), framerate=30).start()
    time.sleep(1)  # Allow camera to warm up

    # FPS calculation
    frame_times = []
    max_frame_history = 30
    enable_motor = not args.no_motor and MOTOR_AVAILABLE

    print("[INFO] Press 'q' or ESC to quit")

    try:
        while True:
            t_start = time.time()
            
            frame = vs.read()
            if frame is None:
                continue

            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            input_data = np.expand_dims(resized, axis=0)

            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
            else:
                # For quantized models, ensure uint8
                if in_dtype == np.uint8 and input_data.dtype != np.uint8:
                    input_data = np.uint8(input_data)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Fetch outputs
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
            
            # Ensure classes are integers
            if classes.dtype not in (np.int32, np.int64):
                classes = classes.astype(np.int32)

            # Draw detections and handle motor control
            target_found = draw_detections_with_tracking(
                frame, boxes, classes, scores, labels, 
                target_class=target_class, 
                thr=args.threshold,
                enable_motor=enable_motor
            )

            # Calculate FPS using moving average
            t_end = time.time()
            frame_time = t_end - t_start
            frame_times.append(frame_time)
            if len(frame_times) > max_frame_history:
                frame_times.pop(0)
            
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Display tracking status
            status = f"Tracking: {target_class} - {'FOUND' if target_found else 'SEARCHING'}"
            cv2.putText(frame, status, (30, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_found else (0, 0, 255), 
                       2, cv2.LINE_AA)

            cv2.imshow("Object Car - TFLite Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        vs.stop()
        cv2.destroyAllWindows()
        # Stop motor if enabled
        if enable_motor:
            try:
                motor.stop()
            except:
                pass

if __name__ == '__main__':
    main()