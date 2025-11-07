#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webcam Object Detection (TFLite)
- Ready for EfficientDet‑Lite0/1/2 (TF Hub) and SSD‑MobileNet (V1/V2/V3) TFLite models
- Works on Raspberry Pi (Pi 4/5) with tflite-runtime or full TensorFlow
- Robust output mapping (auto-detects boxes/classes/scores/count regardless of name/order)
- Keeps your original threaded VideoStream design for better FPS

Usage (examples):
  python3 TFLite_detection_webcam_efficientdet.py \
    --modeldir TFLite_model \
    --graph efficientdet_lite0.tflite \
    --labels labelmap.txt \
    --threshold 0.4 \
    --resolution 1280x720

Notes:
- If you use EfficientDet‑Lite from TF Hub, input size is typically 320x320 (Lite0), 384x384 (Lite1), 448x448 (Lite2).
- This script automatically reads model input tensor size; you can override via --force_input if needed.
- For quantized models (int8/uint8), normalization is not applied; for float models, mean/std normalization is used.
- For EdgeTPU models, add --edgetpu and make sure the corresponding edgetpu .tflite is used.
"""
import os
import sys
import time
import argparse
import importlib.util
from threading import Thread

import cv2
import numpy as np

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

def load_tflite_interpreter(model_path, use_tpu=False, use_xnnpack=True):
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
                break
            except Exception:
                continue

    # XNNPACK CPU acceleration (disabled automatically by some builds)
    # For tflite-runtime >= 2.3 this may be on by default; set via experimental options if supported
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    except TypeError:
        interpreter = Interpreter(model_path=model_path)

    return interpreter

# ------------------------------
# Output mapping (robust to names/order)
# ------------------------------

def map_detection_outputs(output_details):
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
    if n == 4:  # [boxes, classes, scores, count] (TF Hub EfficientDet‑Lite)
        return 0, 1, 2, 3

    raise RuntimeError(f"Unexpected number of outputs: {n}, names={names}")

# ------------------------------
# Drawing helpers
# ------------------------------

def draw_detections(frame, boxes, classes, scores, labels, thr=0.5):
    imH, imW = frame.shape[:2]
    for i in range(len(scores)):
        s = float(scores[i])
        if s < thr or s > 1.0:
            continue
        cls = int(classes[i]) if classes is not None else -1
        # Boxes are usually [ymin, xmin, ymax, xmax] normalized to [0,1]
        y1 = int(max(1,      boxes[i][0] * imH))
        x1 = int(max(1,      boxes[i][1] * imW))
        y2 = int(min(imH-1,  boxes[i][2] * imH))
        x2 = int(min(imW-1,  boxes[i][3] * imW))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 0), 2)
        name = labels[cls] if 0 <= cls < len(labels) else str(cls)
        label = f"{name}: {int(s*100)}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1, th + 10)
        cv2.rectangle(frame, (x1, y_text - th - 8), (x1 + tw + 4, y_text + bl - 8), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (x1 + 2, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modeldir', required=True, help='Folder that contains the .tflite file and labels')
    ap.add_argument('--graph', default='detect.tflite', help='TFLite model filename')
    ap.add_argument('--labels', default='labelmap.txt', help='Label map filename (one class per line)')
    ap.add_argument('--threshold', type=float, default=0.5, help='Minimum confidence threshold')
    ap.add_argument('--resolution', default='640x480', help='Webcam resolution WxH')
    ap.add_argument('--edgetpu', action='store_true', help='Use Coral EdgeTPU delegate')
    ap.add_argument('--camera', type=int, default=0, help='Video device index (default 0)')
    ap.add_argument('--force_input', default='', help='Override model input size as WxH (e.g., 320x320); empty to auto')
    args = ap.parse_args()

    model_path = os.path.join(os.getcwd(), args.modeldir, args.graph)
    labels_path = os.path.join(os.getcwd(), args.modeldir, args.labels)

    # Load labels
    if not os.path.exists(labels_path):
        print(f"[WARN] label file not found: {labels_path}. Using generic labels.")
        labels = [str(i) for i in range(1000)]
    else:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f if l.strip()]
        if labels and labels[0] == '???':
            labels = labels[1:]

    # Load interpreter
    interpreter = load_tflite_interpreter(model_path, use_tpu=args.edgetpu)

    # Determine / set input size
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print outputs for sanity
    print('OUTPUTS =', len(output_details), [d['name'] for d in output_details])

    # Map outputs robustly
    boxes_idx, classes_idx, scores_idx, count_idx = map_detection_outputs(output_details)

    # Figure input shape and dtype
    in_h = int(input_details[0]['shape'][1])
    in_w = int(input_details[0]['shape'][2])
    in_dtype = input_details[0]['dtype']

    if args.force_input:
        try:
            fw, fh = args.force_input.lower().split('x')
            fw, fh = int(fw), int(fh)
            interpreter.resize_tensor_input(input_details[0]['index'], [1, fh, fw, 3])
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            in_h = int(input_details[0]['shape'][1])
            in_w = int(input_details[0]['shape'][2])
            in_dtype = input_details[0]['dtype']
            # Re-map outputs after re-allocate
            boxes_idx, classes_idx, scores_idx, count_idx = map_detection_outputs(output_details)
            print(f"[INFO] Force input size to {fw}x{fh}")
        except Exception as e:
            print(f"[WARN] --force_input ignored due to error: {e}")

    floating_model = (in_dtype == np.float32)
    input_mean, input_std = 127.5, 127.5

    # Prepare camera
    resW, resH = args.resolution.lower().split('x')
    imW, imH = int(resW), int(resH)
    vs = VideoStream(src=args.camera, resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    # FPS calc
    prev_t = time.time()
    fps = 0.0
    freq = cv2.getTickFrequency()
    frame_rate_calc = 1.0

    try:
        while True:
            t1 = cv2.getTickCount()
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
                # For quantized models, uint8 expected
                if input_details[0]['dtype'] == np.uint8 and input_data.dtype != np.uint8:
                    input_data = np.uint8(input_data)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Fetch outputs
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
            if classes.dtype != np.int32 and classes.dtype != np.int64:
                classes = classes.astype(np.int32)

            # Draw
            draw_detections(frame, boxes, classes, scores, labels, thr=args.threshold)

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - prev_t)))
            prev_t = now
            cv2.putText(frame, f"FPS: {frame_rate_calc:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Object detector (TFLite)", frame)
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
