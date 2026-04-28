# #!/home/nvidia/Downloads/surveillance_AI/env/bin/python
# import os
# os.environ["MODEL_CACHE_DIR"] = "/home/nvidia/.cache/roboflow"

# import cv2
# import subprocess
# import time
# import threading
# import signal
# import sys
# import numpy as np
# from inference import get_model

# # ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────
# running = True

# def shutdown(sig, frame):
#     global running
#     print("[INFO] Shutdown signal received, stopping...")
#     running = False

# signal.signal(signal.SIGINT, shutdown)
# signal.signal(signal.SIGTERM, shutdown)
# # ─────────────────────────────────────────────────────────────────────────────

# # ── AUDIO: TTS setup ──────────────────────────────────────────────────────────
# AUDIO_PHRASES = {
#     "person":             "Person detected, please clear the way",
#     #"helmet":             "Worker detected nearby, proceed with caution",
#     #"safety vest":        "Worker in safety vest detected, proceed with caution",
#     #"gloves":             "Worker detected, give way",
#     "forklift":           "Forklift detected, stand clear",
#     "car":                "Vehicle detected ahead, give way",
#     "truck":              "Truck detected, maintaining safe distance",
#     "van":                "Vehicle detected, proceed with caution",
#     "cardboard box":      "Obstacle detected, cardboard box ahead",
#     "wood pallet":        "Pallet detected on path, caution",
#     #"freight container":  "Large container detected, rerouting",
#     #"ladder":             "Ladder detected nearby, caution",
#     "fire":               "Warning! Fire detected, alert personnel immediately",
#     "smoke":              "Warning! Smoke detected, evacuate the area",
#     #"barcode":            "Barcode detected",
#     #"qr code":            "QR code detected",
#     #"license plate":      "License plate detected",
#     #"traffic cone":       "Traffic cone detected, path may be restricted",
#     #"traffic light":      "Traffic light detected ahead",
#     #"road sign":          "Road sign detected",
# }

# COOLDOWNS = {
#     "fire":     4,
#     "smoke":    4,
#     "person":   8,
#     "forklift": 6,
#     "car":      6,
#     "truck":    6,
# }
# DEFAULT_COOLDOWN = 10

# try:
#     import pyttsx3
#     _tts_engine = pyttsx3.init()
#     _tts_engine.setProperty("rate", 145)
#     _tts_engine.setProperty("volume", 1.0)
#     _tts_queue = []
#     _tts_lock = threading.Lock()
#     _cooldown_tracker = {}
#     AUDIO_ENABLED = True

#     def _tts_worker():
#         while True:
#             try:
#                 with _tts_lock:
#                     pending = _tts_queue.copy()
#                     _tts_queue.clear()
#                 if pending:
#                     for phrase in pending:
#                         print(f"[AUDIO] {phrase}")
#                         _tts_engine.say(phrase)
#                     _tts_engine.runAndWait()
#             except Exception as e:
#                 print(f"[AUDIO] TTS error (ignored): {e}")
#             time.sleep(0.2)

#     threading.Thread(target=_tts_worker, daemon=True).start()
#     print("[AUDIO] TTS initialized OK")

# except Exception as e:
#     AUDIO_ENABLED = False
#     print(f"[AUDIO] TTS unavailable, continuing without audio: {e}")

# def announce(class_names):
#     if not AUDIO_ENABLED:
#         return
#     try:
#         now = time.time()
#         to_say = []
#         for name in class_names:
#             cooldown = COOLDOWNS.get(name, DEFAULT_COOLDOWN)
#             if now - _cooldown_tracker.get(name, 0) >= cooldown:
#                 _cooldown_tracker[name] = now
#                 phrase = AUDIO_PHRASES.get(name, f"{name} detected, caution")
#                 to_say.append(phrase)
#         if to_say:
#             with _tts_lock:
#                 _tts_queue.extend(to_say)
#     except Exception as e:
#         print(f"[AUDIO] announce error (ignored): {e}")
# # ── AUDIO END ─────────────────────────────────────────────────────────────────

# # ── COLOUR MAP (BGR) ──────────────────────────────────────────────────────────
# CLASS_COLORS = {
#     "person":            (0, 220,   0),
#     "helmet":            (0, 220, 200),
#     "safety vest":       (0, 200, 180),
#     "gloves":            (0, 180, 160),
#     "car":               (0,   0, 220),
#     "truck":             (0,   0, 220),
#     "van":               (0,   0, 220),
#     "forklift":          (0,   0, 255),
#     "cardboard box":     (0, 165, 255),
#     "wood pallet":       (0, 140, 230),
#     "freight container": (0, 120, 210),
#     "ladder":            (200,  0, 160),
#     "traffic cone":      (0, 220, 255),
#     "traffic light":     (0, 230, 255),
#     "road sign":         (0, 240, 255),
#     "fire":              (0,  80, 255),
#     "smoke":             (30, 140, 255),
#     "barcode":           (220,  0, 220),
#     "qr code":           (200,  0, 200),
#     "license plate":     (180,  0, 180),
# }
# DEFAULT_COLOR = (160, 160, 160)
# # ─────────────────────────────────────────────────────────────────────────────

# API_KEY = "mLJy64m9DaOxKTiSviMh"

# # FIX 1: subtype=01 → H.264 substream, much cheaper for OpenCV to decode than H.265
# RTSP_INPUT  = "rtsp://admin:trov%40123@192.168.1.161:554/cam/realmonitor?channel=1&subtype=01"
# RTSP_OUTPUT = "rtsp://localhost:8554/ai_surveillance"

# TARGET_FPS     = 5
# FRAME_INTERVAL = 1.0 / TARGET_FPS

# # FIX 3: run inference every 4th frame instead of every 2nd
# # reduces inference load by 2x while bounding boxes still feel smooth
# INFER_EVERY_N = 4

# model = get_model(model_id="logistics-sz9jr/2", api_key=API_KEY)

# def open_capture(url, retries=5):
#     for attempt in range(retries):
#         if not running:
#             return None
#         cap = cv2.VideoCapture(url)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         if cap.isOpened():
#             print(f"[INFO] Camera connected on attempt {attempt + 1}")
#             return cap
#         print(f"[WARN] Camera open failed (attempt {attempt + 1}/{retries}), retrying in 2s...")
#         time.sleep(2)
#     print("[ERROR] Could not open camera after retries.")
#     return None

# # FIX 2: h264_v4l2m2m = Jetson hardware VPU encoder, offloads encoding from CPU
# # libx264 was doing software encoding on CPU — that alone was eating 2-3 cores
# # h264_v4l2m2m failed — no valid V4L2 device accessible via pipe on this Jetson build
# # Back to libx264 ultrafast with lower bitrate + loglevel error to suppress ffmpeg spam
# def open_ffmpeg():
#     return subprocess.Popen([
#         "ffmpeg", "-loglevel", "error",
#         "-f", "rawvideo", "-pixel_format", "bgr24",
#         "-video_size", "1280x720", "-r", str(TARGET_FPS),
#         "-i", "-",
#         "-c:v", "libx264",
#         "-preset", "ultrafast",
#         "-tune", "zerolatency",
#         "-g", str(TARGET_FPS),
#         "-b:v", "1000k", "-maxrate", "1000k",
#         "-f", "rtsp", RTSP_OUTPUT
#     ], stdin=subprocess.PIPE)

# cap  = open_capture(RTSP_INPUT)
# proc = open_ffmpeg()

# frame_count          = 0
# boxes                = []
# consecutive_failures = 0
# MAX_FAILURES         = 30

# print("[INFO] Starting main loop...")
# print(f"[INFO] Input:  {RTSP_INPUT}")
# print(f"[INFO] Output: {RTSP_OUTPUT}")
# print(f"[INFO] Inference every {INFER_EVERY_N} frames")
# print(f"[INFO] Encoder: h264_v4l2m2m (hardware VPU)")

# while running:
#     loop_start = time.time()

#     if cap is None or not cap.isOpened():
#         print("[WARN] Camera not available, attempting reconnect...")
#         if cap:
#             cap.release()
#         time.sleep(2)
#         cap = open_capture(RTSP_INPUT)
#         consecutive_failures = 0
#         continue

#     ret, frame = cap.read()

#     if not ret:
#         consecutive_failures += 1
#         print(f"[WARN] Frame read failed ({consecutive_failures}/{MAX_FAILURES})")
#         if consecutive_failures >= MAX_FAILURES:
#             print("[WARN] Too many failures, reconnecting camera...")
#             cap.release()
#             cap = open_capture(RTSP_INPUT)
#             consecutive_failures = 0
#         time.sleep(0.1)
#         continue

#     consecutive_failures = 0
#     frame = cv2.resize(frame, (1280, 720))
#     frame_count += 1

#     # FIX 3: infer every 4th frame
#     if frame_count % INFER_EVERY_N == 0:
#         try:
#             results = model.infer(frame, confidence=0.5)[0]
#             boxes = results.predictions
#             if boxes:
#                 announce(list({p.class_name for p in boxes}))
#         except Exception as e:
#             print(f"[WARN] Inference error: {e}")

#     for pred in boxes[-5:]:
#         x1 = max(0, int(pred.x - pred.width / 2))
#         y1 = max(0, int(pred.y - pred.height / 2))
#         x2 = min(1280, int(pred.x + pred.width / 2))
#         y2 = min(720,  int(pred.y + pred.height / 2))
#         color = CLASS_COLORS.get(pred.class_name.lower(), DEFAULT_COLOR)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, pred.class_name, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     try:
#         proc.stdin.write(frame.tobytes())
#     except BrokenPipeError:
#         if not running:
#             break
#         print("[WARN] ffmpeg pipe broken, restarting ffmpeg...")
#         proc.kill()
#         proc = open_ffmpeg()
#         continue

#     elapsed    = time.time() - loop_start
#     sleep_time = FRAME_INTERVAL - elapsed
#     if sleep_time > 0:
#         time.sleep(sleep_time)

# # ── CLEANUP ───────────────────────────────────────────────────────────────────
# print("[INFO] Cleaning up...")
# if cap:
#     cap.release()
# if proc:
#     proc.kill()
# print("[INFO] Done.")
# sys.exit(0)






























# #!/home/nvidia/Downloads/surveillance_AI/env/bin/python
# import os
# os.environ["MODEL_CACHE_DIR"] = "/home/nvidia/.cache/roboflow"

# import cv2
# import subprocess
# import time
# import threading
# import signal
# import sys
# import numpy as np
# import fcntl
# from ultralytics import YOLO

# # ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────
# running = True

# def shutdown(sig, frame):
#     global running
#     print("[INFO] Shutdown signal received, stopping...")
#     running = False

# signal.signal(signal.SIGINT, shutdown)
# signal.signal(signal.SIGTERM, shutdown)
# # ─────────────────────────────────────────────────────────────────────────────

# # ── AUDIO: TTS setup ──────────────────────────────────────────────────────────
# AUDIO_PHRASES = {
#     "person":             "Person detected, please clear the way",
#     "forklift":           "Forklift detected, stand clear",
#     "car":                "Vehicle detected ahead, give way",
#     "truck":              "Truck detected, maintaining safe distance",
#     "van":                "Vehicle detected, proceed with caution",
#     "cardboard box":      "Obstacle detected, cardboard box ahead",
#     "wood pallet":        "Pallet detected on path, caution",
#     "fire":               "Warning! Fire detected, alert personnel immediately",
#     "smoke":              "Warning! Smoke detected, evacuate the area",
# }

# COOLDOWNS = {
#     "fire":     4,
#     "smoke":    4,
#     "person":   8,
#     "forklift": 6,
#     "car":      6,
#     "truck":    6,
# }
# DEFAULT_COOLDOWN = 10

# try:
#     import pyttsx3
#     _tts_engine = pyttsx3.init()
#     _tts_engine.setProperty("rate", 145)
#     _tts_engine.setProperty("volume", 1.0)
#     _tts_queue = []
#     _tts_lock = threading.Lock()
#     _cooldown_tracker = {}
#     AUDIO_ENABLED = True

#     def _tts_worker():
#         while True:
#             try:
#                 with _tts_lock:
#                     pending = _tts_queue.copy()
#                     _tts_queue.clear()
#                 if pending:
#                     for phrase in pending:
#                         print(f"[AUDIO] {phrase}")
#                         _tts_engine.say(phrase)
#                     _tts_engine.runAndWait()
#             except Exception as e:
#                 print(f"[AUDIO] TTS error (ignored): {e}")
#             time.sleep(0.2)

#     threading.Thread(target=_tts_worker, daemon=True).start()
#     print("[AUDIO] TTS initialized OK")

# except Exception as e:
#     AUDIO_ENABLED = False
#     print(f"[AUDIO] TTS unavailable, continuing without audio: {e}")

# def announce(class_names):
#     if not AUDIO_ENABLED:
#         return
#     try:
#         now = time.time()
#         to_say = []
#         for name in class_names:
#             cooldown = COOLDOWNS.get(name, DEFAULT_COOLDOWN)
#             if now - _cooldown_tracker.get(name, 0) >= cooldown:
#                 _cooldown_tracker[name] = now
#                 phrase = AUDIO_PHRASES.get(name, f"{name} detected, caution")
#                 to_say.append(phrase)
#         if to_say:
#             with _tts_lock:
#                 _tts_queue.extend(to_say)
#     except Exception as e:
#         print(f"[AUDIO] announce error (ignored): {e}")
# # ── AUDIO END ─────────────────────────────────────────────────────────────────

# # ── COLOUR MAP (BGR) ──────────────────────────────────────────────────────────
# CLASS_COLORS = {
#     "person":            (0, 220,   0),
#     "helmet":            (0, 220, 200),
#     "safety vest":       (0, 200, 180),
#     "gloves":            (0, 180, 160),
#     "car":               (0,   0, 220),
#     "truck":             (0,   0, 220),
#     "van":               (0,   0, 220),
#     "forklift":          (0,   0, 255),
#     "cardboard box":     (0, 165, 255),
#     "wood pallet":       (0, 140, 230),
#     "freight container": (0, 120, 210),
#     "ladder":            (200,  0, 160),
#     "traffic cone":      (0, 220, 255),
#     "traffic light":     (0, 230, 255),
#     "road sign":         (0, 240, 255),
#     "fire":              (0,  80, 255),
#     "smoke":             (30, 140, 255),
#     "barcode":           (220,  0, 220),
#     "qr code":           (200,  0, 200),
#     "license plate":     (180,  0, 180),
# }
# DEFAULT_COLOR = (160, 160, 160)
# # ─────────────────────────────────────────────────────────────────────────────

# ENGINE_PATH    = "/home/nvidia/.cache/roboflow/logistics-sz9jr/2/weights.engine"
# RTSP_INPUT     = "rtsp://admin:trov%40123@192.168.1.161:554/cam/realmonitor?channel=1&subtype=01"
# RTSP_OUTPUT    = "rtsp://localhost:8554/ai_surveillance"
# TARGET_FPS     = 15
# FRAME_INTERVAL = 1.0 / TARGET_FPS
# INFER_EVERY_N  = 1  # infer every frame — GPU is fast enough

# model = YOLO(ENGINE_PATH, task="detect")

# # Load class names — TensorRT engine doesn't store them
# with open("/home/nvidia/.cache/roboflow/logistics-sz9jr/2/class_names.txt") as f:
#     CLASS_NAMES = {i: line.strip() for i, line in enumerate(f.readlines())}

# print(f"[INFO] Loaded {len(CLASS_NAMES)} class names: {list(CLASS_NAMES.values())}")


# def open_capture(url, retries=5):
#     for attempt in range(retries):
#         if not running:
#             return None
#         cap = cv2.VideoCapture(url)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         if cap.isOpened():
#             print(f"[INFO] Camera connected on attempt {attempt + 1}")
#             return cap
#         print(f"[WARN] Camera open failed (attempt {attempt + 1}/{retries}), retrying in 2s...")
#         time.sleep(2)
#     print("[ERROR] Could not open camera after retries.")
#     return None


# def open_ffmpeg():
#     proc = subprocess.Popen([
#         "ffmpeg", "-loglevel", "error",
#         "-f", "rawvideo", "-pixel_format", "bgr24",
#         "-video_size", "640x480", "-r", str(TARGET_FPS),
#         "-i", "-",
#         "-c:v", "libx264",
#         "-preset", "ultrafast",
#         "-tune", "zerolatency",
#         "-g", str(TARGET_FPS),
#         "-b:v", "1000k", "-maxrate", "1000k",
#         "-f", "rtsp", RTSP_OUTPUT
#     ], stdin=subprocess.PIPE)

#     # Set pipe to non-blocking so frames are dropped instead of queued
#     # This prevents the multi-minute delay that builds up over time
#     fd = proc.stdin.fileno()
#     flags = fcntl.fcntl(fd, fcntl.F_GETFL)
#     fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

#     return proc


# cap  = open_capture(RTSP_INPUT)
# proc = open_ffmpeg()

# frame_count          = 0
# boxes                = []
# consecutive_failures = 0
# MAX_FAILURES         = 30
# frames_dropped       = 0

# print("[INFO] Starting main loop...")
# print(f"[INFO] Input:  {RTSP_INPUT}")
# print(f"[INFO] Output: {RTSP_OUTPUT}")
# print(f"[INFO] Inference every frame (GPU TensorRT)")

# while running:
#     loop_start = time.time()

#     if cap is None or not cap.isOpened():
#         print("[WARN] Camera not available, attempting reconnect...")
#         if cap:
#             cap.release()
#         time.sleep(2)
#         cap = open_capture(RTSP_INPUT)
#         consecutive_failures = 0
#         continue

#     ret, frame = cap.read()

#     if not ret:
#         consecutive_failures += 1
#         print(f"[WARN] Frame read failed ({consecutive_failures}/{MAX_FAILURES})")
#         if consecutive_failures >= MAX_FAILURES:
#             print("[WARN] Too many failures, reconnecting camera...")
#             cap.release()
#             cap = open_capture(RTSP_INPUT)
#             consecutive_failures = 0
#         time.sleep(0.1)
#         continue

#     consecutive_failures = 0
#     frame = cv2.resize(frame, (640, 480))
#     frame_count += 1

#     if frame_count % INFER_EVERY_N == 0:
#         try:
#             results = model(frame, device=0, verbose=False, conf=0.5)[0]
#             boxes = results.boxes
#             if boxes is not None and len(boxes):
#                 announce(list({CLASS_NAMES[int(c)] for c in boxes.cls}))
#         except Exception as e:
#             print(f"[WARN] Inference error: {e}")

#     for box in (boxes[-5:] if boxes is not None and len(boxes) else []):
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(640, x2), min(480, y2)
#         class_name = CLASS_NAMES[int(box.cls[0])]
#         color = CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, class_name, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     try:
#         proc.stdin.write(frame.tobytes())
#     except BlockingIOError:
#         # ffmpeg is busy — drop this frame, don't queue it
#         frames_dropped += 1
#         if frames_dropped % 50 == 0:
#             print(f"[WARN] Dropped {frames_dropped} frames total (ffmpeg busy)")
#     except BrokenPipeError:
#         if not running:
#             break
#         print("[WARN] ffmpeg pipe broken, restarting ffmpeg...")
#         proc.kill()
#         proc = open_ffmpeg()
#         continue

#     elapsed    = time.time() - loop_start
#     sleep_time = FRAME_INTERVAL - elapsed
#     if sleep_time > 0:
#         time.sleep(sleep_time)

# # ── CLEANUP ───────────────────────────────────────────────────────────────────
# print("[INFO] Cleaning up...")
# if cap:
#     cap.release()
# if proc:
#     proc.kill()
# print("[INFO] Done.")
# sys.exit(0)










#!/home/nvidia/Downloads/surveillance_AI/env/bin/python
import os
os.environ["MODEL_CACHE_DIR"] = "/home/nvidia/.cache/roboflow"

import cv2
import subprocess
import time
import threading
import signal
import sys
import fcntl
import numpy as np
from ultralytics import YOLO

# ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────
running = True

def shutdown(sig, frame):
    global running
    print("[INFO] Shutdown signal received, stopping...")
    running = False

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)
# ─────────────────────────────────────────────────────────────────────────────

# ── AUDIO: TTS setup ──────────────────────────────────────────────────────────
AUDIO_PHRASES = {
    "person":             "Person detected, please clear the way",
    "forklift":           "Forklift detected, stand clear",
    "car":                "Vehicle detected ahead, give way",
    "truck":              "Truck detected, maintaining safe distance",
    "van":                "Vehicle detected, proceed with caution",
    "cardboard box":      "Obstacle detected, cardboard box ahead",
    "wood pallet":        "Pallet detected on path, caution",
    "fire":               "Warning! Fire detected, alert personnel immediately",
    "smoke":              "Warning! Smoke detected, evacuate the area",
}

COOLDOWNS = {
    "fire":     4,
    "smoke":    4,
    "person":   8,
    "forklift": 6,
    "car":      6,
    "truck":    6,
}
DEFAULT_COOLDOWN = 10

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 145)
    _tts_engine.setProperty("volume", 1.0)
    _tts_queue = []
    _tts_lock = threading.Lock()
    _cooldown_tracker = {}
    AUDIO_ENABLED = True

    def _tts_worker():
        while True:
            try:
                with _tts_lock:
                    pending = _tts_queue.copy()
                    _tts_queue.clear()
                if pending:
                    for phrase in pending:
                        print(f"[AUDIO] {phrase}")
                        _tts_engine.say(phrase)
                    _tts_engine.runAndWait()
            except Exception as e:
                print(f"[AUDIO] TTS error (ignored): {e}")
            time.sleep(0.2)

    threading.Thread(target=_tts_worker, daemon=True).start()
    print("[AUDIO] TTS initialized OK")

except Exception as e:
    AUDIO_ENABLED = False
    print(f"[AUDIO] TTS unavailable, continuing without audio: {e}")

def announce(class_names):
    if not AUDIO_ENABLED:
        return
    try:
        now = time.time()
        to_say = []
        for name in class_names:
            cooldown = COOLDOWNS.get(name, DEFAULT_COOLDOWN)
            if now - _cooldown_tracker.get(name, 0) >= cooldown:
                _cooldown_tracker[name] = now
                phrase = AUDIO_PHRASES.get(name, f"{name} detected, caution")
                to_say.append(phrase)
        if to_say:
            with _tts_lock:
                _tts_queue.extend(to_say)
    except Exception as e:
        print(f"[AUDIO] announce error (ignored): {e}")
# ── AUDIO END ─────────────────────────────────────────────────────────────────

# ── COLOUR MAP (BGR) ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "person":            (0, 220,   0),
    "helmet":            (0, 220, 200),
    "safety vest":       (0, 200, 180),
    "gloves":            (0, 180, 160),
    "car":               (0,   0, 220),
    "truck":             (0,   0, 220),
    "van":               (0,   0, 220),
    "forklift":          (0,   0, 255),
    "cardboard box":     (0, 165, 255),
    "wood pallet":       (0, 140, 230),
    "freight container": (0, 120, 210),
    "ladder":            (200,  0, 160),
    "traffic cone":      (0, 220, 255),
    "traffic light":     (0, 230, 255),
    "road sign":         (0, 240, 255),
    "fire":              (0,  80, 255),
    "smoke":             (30, 140, 255),
    "barcode":           (220,  0, 220),
    "qr code":           (200,  0, 200),
    "license plate":     (180,  0, 180),
}
DEFAULT_COLOR = (160, 160, 160)
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_PATH    = "/home/nvidia/.cache/roboflow/logistics-sz9jr/2/weights.engine"
RTSP_INPUT     = "rtsp://admin:trov%40123@192.168.1.161:554/cam/realmonitor?channel=1&subtype=00"
RTSP_OUTPUT    = "rtsp://localhost:8554/ai_surveillance"

# ── FPS STRATEGY ──────────────────────────────────────────────────────────────
# Camera delivers ~34 FPS. We cap reads at TARGET_FPS by throttling in the loop.
# ffmpeg is given a steady TARGET_FPS stream → no mid-GOP drops → no tile artifact.
TARGET_FPS     = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Run inference on every frame — TensorRT on Orin GPU is fast enough at 10 FPS
INFER_EVERY_N  = 1
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO(ENGINE_PATH, task="detect")

with open("/home/nvidia/.cache/roboflow/logistics-sz9jr/2/class_names.txt") as f:
    CLASS_NAMES = {i: line.strip() for i, line in enumerate(f.readlines())}

print(f"[INFO] Loaded {len(CLASS_NAMES)} class names: {list(CLASS_NAMES.values())}")


def open_capture(url, retries=5):
    for attempt in range(retries):
        if not running:
            return None
        cap = cv2.VideoCapture(url)
        # Buffer size 1: always grab the newest frame, discard stale ones
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"[INFO] Camera connected on attempt {attempt + 1}")
            return cap
        print(f"[WARN] Camera open failed (attempt {attempt + 1}/{retries}), retrying in 2s...")
        time.sleep(2)
    print("[ERROR] Could not open camera after retries.")
    return None


def open_ffmpeg():
    """
    Key changes vs previous version:
      - BLOCKING pipe (no O_NONBLOCK) — we control send rate via FRAME_INTERVAL,
        so ffmpeg is never overwhelmed and the pipe never needs to drop frames.
      - keyint=20:min-keyint=20 forces a keyframe every 2 s at 10 FPS,
        giving HLS clean segment boundaries and eliminating the tile artifact.
      - bufsize = 2× maxrate for a smoother encoder buffer.
    """
    proc = subprocess.Popen([
        "ffmpeg", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "640x480", "-r", str(TARGET_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-x264-params", f"keyint={TARGET_FPS * 2}:min-keyint={TARGET_FPS * 2}",
        "-b:v", "600k", "-maxrate", "600k", "-bufsize", "1200k",
        "-f", "rtsp", RTSP_OUTPUT
    ], stdin=subprocess.PIPE)
    # NOTE: pipe is intentionally left BLOCKING.
    # The loop sleeps for FRAME_INTERVAL each iteration, so ffmpeg is never
    # sent frames faster than it can encode them.
    return proc


cap  = open_capture(RTSP_INPUT)
proc = open_ffmpeg()

frame_count          = 0
boxes                = []
consecutive_failures = 0
MAX_FAILURES         = 30

print("[INFO] Starting main loop...")
print(f"[INFO] Input:  {RTSP_INPUT}")
print(f"[INFO] Output: {RTSP_OUTPUT}")
print(f"[INFO] Target FPS: {TARGET_FPS}  |  Inference every {INFER_EVERY_N} frame(s)")

while running:
    loop_start = time.time()

    # ── Camera reconnect ──────────────────────────────────────────────────────
    if cap is None or not cap.isOpened():
        print("[WARN] Camera not available, attempting reconnect...")
        if cap:
            cap.release()
        time.sleep(2)
        cap = open_capture(RTSP_INPUT)
        consecutive_failures = 0
        continue

    # ── Drain the camera buffer so we always get the latest frame ─────────────
    # At 34 FPS input and 10 FPS output, OpenCV queues up ~3 stale frames.
    # Grab (decode-less) all but the last, then do one full read.
    for _ in range(2):
        cap.grab()
    ret, frame = cap.read()

    if not ret:
        consecutive_failures += 1
        print(f"[WARN] Frame read failed ({consecutive_failures}/{MAX_FAILURES})")
        if consecutive_failures >= MAX_FAILURES:
            print("[WARN] Too many failures, reconnecting camera...")
            cap.release()
            cap = open_capture(RTSP_INPUT)
            consecutive_failures = 0
        time.sleep(0.1)
        continue

    consecutive_failures = 0
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # ── Inference ─────────────────────────────────────────────────────────────
    if frame_count % INFER_EVERY_N == 0:
        try:
            results = model(frame, device=0, verbose=False, conf=0.5)[0]
            boxes = results.boxes
            if boxes is not None and len(boxes):
                announce(list({CLASS_NAMES[int(c)] for c in boxes.cls}))
        except Exception as e:
            print(f"[WARN] Inference error: {e}")

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    for box in (boxes[-5:] if boxes is not None and len(boxes) else []):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(640, x2), min(480, y2)
        class_name = CLASS_NAMES[int(box.cls[0])]
        conf_pct   = int(box.conf[0] * 100)
        color      = CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {conf_pct}%", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Send to ffmpeg (blocking — pipe paced by FRAME_INTERVAL sleep below) ──
    try:
        proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        if not running:
            break
        print("[WARN] ffmpeg pipe broken, restarting ffmpeg...")
        proc.kill()
        proc = open_ffmpeg()
        continue

    # ── Pace the loop to TARGET_FPS ───────────────────────────────────────────
    elapsed    = time.time() - loop_start
    sleep_time = FRAME_INTERVAL - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

# ── CLEANUP ───────────────────────────────────────────────────────────────────
print("[INFO] Cleaning up...")
if cap:
    cap.release()
if proc:
    proc.stdin.close()
    proc.wait()
print("[INFO] Done.")
sys.exit(0)