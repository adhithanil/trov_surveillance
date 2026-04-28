# TROV Surveillance AI

Real-time object detection for the TROV robot's surveillance camera feed. Runs entirely on the Jetson AGX Orin — no cloud, no API calls, no internet dependency. A TensorRT-optimised YOLOv8 model detects people, vehicles, and warehouse hazards in the camera stream, draws annotated bounding boxes, announces detections through a speaker via TTS, and pushes the annotated stream back out to MediaMTX so the web UI can display it live.

---

## How it fits into the TROV system

MediaMTX is configured to run `start_ai.sh` automatically when the `ai_surveillance` RTSP path is requested. The script activates the Python virtual environment and starts `ai_publish.py`. The annotated output stream is then available at `rtsp://localhost:8554/ai_surveillance` and the web UI picks it up from there.

```
IMOU surveillance camera  (192.168.1.161)
    ↓  RTSP stream
ai_publish.py  (Jetson GPU — TensorRT inference)
    ↓  annotated frames via ffmpeg pipe
MediaMTX  (rtsp://localhost:8554/ai_surveillance)
    ↓  RTSP / WebRTC
Web UI  (surveillance camera view)
```

---

## What it detects

The model is Roboflow's pretrained logistics dataset (`logistics-sz9jr/2`), exported to TensorRT for the Jetson. It detects warehouse-relevant objects including people, forklifts, cars, trucks, vans, cardboard boxes, wood pallets, freight containers, ladders, traffic cones, fire, smoke, barcodes, QR codes, and license plates.

Each detected class has a colour-coded bounding box:

| Colour | Classes |
|---|---|
| Green | Person |
| Blue | Car, truck, van |
| Bright blue | Forklift |
| Orange | Cardboard box, wood pallet |
| Red/orange | Fire, smoke |
| Purple | Barcodes, QR codes, license plates |
| Grey | Everything else |

Confidence percentage is shown next to each label.

---

## Audio announcements

When a detection occurs, the system speaks an announcement through the Jetson's audio output using `pyttsx3` (local TTS, no internet). Each class has a cooldown timer to prevent the same announcement from repeating too frequently:

| Class | Cooldown | Announcement |
|---|---|---|
| Person | 8s | "Person detected, please clear the way" |
| Forklift | 6s | "Forklift detected, stand clear" |
| Fire | 4s | "Warning! Fire detected, alert personnel immediately" |
| Smoke | 4s | "Warning! Smoke detected, evacuate the area" |
| Car / Truck | 6s | "Vehicle detected ahead, give way" |
| Others | 10s | "[class name] detected, caution" |

TTS runs in a background thread so it never blocks the video pipeline.

---

## Technical details

The main loop reads frames from the IMOU camera over RTSP, drains the camera buffer on each iteration (two `grab()` calls before `read()`) to always get the freshest frame rather than a stale buffered one. Frames are resized to 640×480 and passed to the TensorRT engine via Ultralytics YOLO. Inference runs on every frame — the Jetson Orin GPU handles this comfortably at 25 FPS.

Annotated frames are piped to ffmpeg, which encodes them as H.264 and pushes the stream to MediaMTX over RTSP. The ffmpeg pipe is intentionally left blocking — the loop sleeps for `1/TARGET_FPS` seconds at the end of each iteration, so ffmpeg is never sent frames faster than it can encode them. Keyframes are forced every 2 seconds (`keyint = TARGET_FPS × 2`) to give HLS clean segment boundaries.

The TTS engine runs in a separate daemon thread with a queue and lock. Announcements are batched per detection cycle and played sequentially without blocking the video loop.

---

## Files

| File | Purpose |
|---|---|
| `ai_publish.py` | Main script — inference, annotation, audio, ffmpeg output |
| `final_live_detection.py` | Earlier version using Roboflow cloud API — not used |
| `start_ai.sh` | Entry point called by MediaMTX — activates venv and runs `ai_publish.py` |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (not tracked — contains API key from older version) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/adhith-edgeforce/trov_surveillance.git
cd trov_surveillance
```

### 2. Create the virtual environment

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Export the TensorRT engine

The model must be exported to TensorRT on the Jetson itself — TensorRT engines are not portable between machines. Download the Roboflow logistics model weights and export:

```bash
from ultralytics import YOLO
model = YOLO("path/to/logistics-sz9jr-2.pt")
model.export(format="engine", device=0, half=True)
```

Place the exported `.engine` file and `class_names.txt` at:
```
/home/nvidia/.cache/roboflow/logistics-sz9jr/2/weights.engine
/home/nvidia/.cache/roboflow/logistics-sz9jr/2/class_names.txt
```

### 4. Configure the camera

Edit `ai_publish.py` and update `RTSP_INPUT` to point to your camera:

```python
RTSP_INPUT = "rtsp://admin:PASSWORD@YOUR_CAMERA_IP:554/cam/realmonitor?channel=1&subtype=00"
```

### 5. Run manually

```bash
source env/bin/activate
python ai_publish.py
```

### 6. Run via MediaMTX (normal operation)

Place `start_ai.sh` where MediaMTX expects it (configured in `mediamtx.yml` under the `ai_surveillance` path). MediaMTX will call it automatically when the stream is requested.

---

## Dependencies

- `ultralytics` — YOLOv8 inference
- `opencv-python` — frame capture and annotation
- `pyttsx3` — local TTS audio
- `numpy`
- TensorRT (installed as part of JetPack on Jetson)
- ffmpeg (system package)
- MediaMTX (see main TROV repo)
