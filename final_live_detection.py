"""
Robot Vision System - Final Live Detection
==========================================
Uses Roboflow Inference (local) with logistics-sz9jr/2 model.
API key used ONCE to download weights — then 100% offline.

Detected classes (20):
  barcode, car, cardboard box, fire, forklift, freight container,
  gloves, helmet, ladder, license plate, person, qr code, road sign,
  safety vest, smoke, traffic cone, traffic light, truck, van, wood pallet

Controls:  Q=quit  S=screenshot  D=detect now  F=toggle interval
Version: 4.0.0  (RTSP + NumPy 2.x compatible)

RTSP notes:
  - Set CAMERA_SOURCE=rtsp://user:pass@192.168.x.x/stream in .env
  - Set RTSP_TRANSPORT=tcp (default) or udp
  - Auto-reconnects on stream drop
  - Dedicated reader thread eliminates buffer lag
"""

import cv2
import os
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env BEFORE importing anything else
# ---------------------------------------------------------------------------
# Look for .env in the same directory as this script
_script_dir = Path(__file__).parent.resolve()
_env_path   = _script_dir / ".env"

if _env_path.exists():
    # Manual parse — avoids python-dotenv issues on some setups
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")   # remove any quotes
            os.environ.setdefault(_k, _v)
    print(f"✅  Loaded .env from: {_env_path}")
else:
    print(f"⚠️   No .env found at: {_env_path}")
    print("    Create .env in the same folder as this script.")

# ---------------------------------------------------------------------------
# NumPy version guard — must be before any numpy-dependent import
# ---------------------------------------------------------------------------
try:
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(f"\n❌  NumPy {np.__version__} detected — incompatible with onnxruntime.")
        print("    Fix:  pip install \"numpy<2\"")
        print("    Then re-run the script.\n")
        sys.exit(1)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Suppress noisy TF / timm warnings before heavy imports
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Roboflow Inference import
# ---------------------------------------------------------------------------
try:
    from inference import get_model
except Exception as exc:
    print(f"\n❌  Failed to import 'inference': {exc}")
    print("    Try:  pip install inference")
    print("    If NumPy error persists:  pip install \"numpy<2\"")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colour map  (BGR)
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "person":             (0,  220,   0),   # green
    "car":                (0,    0, 220),   # red
    "truck":              (0,    0, 220),
    "van":                (0,    0, 220),
    "forklift":           (0,    0, 255),
    "fire":               (0,  100, 255),   # orange-red
    "smoke":              (30, 140, 255),
    "helmet":             (0,  220, 200),   # teal
    "safety vest":        (0,  200, 180),
    "gloves":             (0,  180, 160),
    "cardboard box":      (0,  165, 255),   # orange
    "wood pallet":        (0,  140, 230),
    "freight container":  (0,  120, 210),
    "barcode":            (220,  0, 220),   # magenta
    "qr code":            (200,  0, 200),
    "license plate":      (180,  0, 180),
    "traffic cone":       (0,  220, 255),   # yellow
    "traffic light":      (0,  230, 255),
    "road sign":          (0,  240, 255),
    "ladder":             (200,  0, 160),   # purple
}
DEFAULT_COLOR = (160, 160, 160)


class RobotVisionSystem:
    """Real-time local inference using Roboflow logistics-sz9jr/2 model."""

    MODEL_ID = "logistics-sz9jr/2"

    def __init__(self):
        # ── API key ──────────────────────────────────────────────────────
        self.api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'")

        if not self.api_key or self.api_key == "your_api_key_here":
            print("\n❌  ROBOFLOW_API_KEY is not set properly.")
            print(f"    .env path checked: {_env_path}")
            print("    Make sure your .env contains:")
            print("        ROBOFLOW_API_KEY=rf_yourActualKeyHere")
            print("    (no quotes, no spaces around the '=')\n")
            sys.exit(1)

        self.confidence = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

        # ── Load model ───────────────────────────────────────────────────
        print(f"📦  Loading model: {self.MODEL_ID}")
        print("    (first run downloads ~50 MB — subsequent runs are instant)\n")
        self.model = get_model(model_id=self.MODEL_ID, api_key=self.api_key)
        print("\n✅  Model ready — running fully offline from now on!\n")

        # ── Camera ───────────────────────────────────────────────────────
        src = os.environ.get("CAMERA_SOURCE", "0").strip()
        try:
            self.cam_src = int(src)
            self.is_rtsp = False
        except ValueError:
            self.cam_src = src
            self.is_rtsp = src.lower().startswith("rtsp://")

        # RTSP transport protocol: tcp (reliable) or udp (lower latency)
        self.rtsp_transport    = os.environ.get("RTSP_TRANSPORT", "tcp").lower()
        self.reconnect_delay   = float(os.environ.get("RTSP_RECONNECT_DELAY", "3.0"))
        self.reconnect_attempts = 0

        # Frame queue: reader thread pushes, main thread pops latest
        # maxsize=1 means we always get the freshest frame (no buffer lag)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._cap_lock = threading.Lock()

        # ── Detection state ──────────────────────────────────────────────
        self.detections:  list  = []
        self.lock                = threading.Lock()
        self.detecting:   bool  = False
        self.last_detect: float = 0.0
        self.interval:    float = float(os.environ.get("DETECTION_INTERVAL", "0.5"))

        # ── Stream state ─────────────────────────────────────────────────
        self.stream_ok:   bool  = False   # True once first frame received

        # ── Perf counters ────────────────────────────────────────────────
        self.frame_count      = 0
        self.total_detections = 0
        self.fps              = 0.0
        self._fps_t           = time.time()
        self._fps_n           = 0
        self.running          = True

    # ── RTSP helpers ───────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Open the camera / RTSP stream with correct settings.
        For RTSP: forces FFMPEG backend, TCP transport, and minimal buffering
        so we always get the latest frame with no lag.
        """
        if self.is_rtsp:
            # Tell FFMPEG to use TCP (more reliable than UDP for most cameras)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|"
                "stimeout;5000000|"          # 5 s connection timeout
                "buffer_size;65536"          # small network buffer
            )
            cap = cv2.VideoCapture(self.cam_src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only 1 frame in buffer
        else:
            cap = cv2.VideoCapture(self.cam_src)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _frame_reader(self, cap: cv2.VideoCapture) -> None:
        """
        Dedicated thread that continuously grabs frames from the capture device.
        For RTSP this is critical: it drains the network buffer so the main
        thread always gets the most recent frame rather than a stale queued one.
        On failure it sets a sentinel (None) so the main loop can reconnect.
        """
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Signal failure to main thread
                try:
                    self._frame_queue.put_nowait(None)
                except queue.Full:
                    pass
                break

            # Overwrite any unread frame — we only ever want the latest
            try:
                self._frame_queue.get_nowait()   # discard old frame if present
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)
            self.stream_ok = True

    # ── Inference (background thread) ─────────────────────────────────────

    def _infer(self, frame) -> None:
        try:
            results = self.model.infer(image=frame, confidence=self.confidence)[0]
            preds = []
            for pred in results.predictions:
                preds.append({
                    "class":      pred.class_name,
                    "confidence": pred.confidence,
                    "x1": int(pred.x - pred.width  / 2),
                    "y1": int(pred.y - pred.height / 2),
                    "x2": int(pred.x + pred.width  / 2),
                    "y2": int(pred.y + pred.height / 2),
                })

            with self.lock:
                self.detections        = preds
                self.total_detections += len(preds)

            if preds:
                names = ", ".join(p["class"] for p in preds[:6])
                extra = f" +{len(preds)-6} more" if len(preds) > 6 else ""
                print(f"🎯  {len(preds)} object(s): {names}{extra}")
            else:
                print("🔍  Nothing detected this frame")

        except Exception as exc:
            print(f"⚠️   Inference error: {exc}")
        finally:
            self.detecting = False

    # ── Drawing ────────────────────────────────────────────────────────────

    @staticmethod
    def _color(name: str) -> tuple:
        return CLASS_COLORS.get(name.lower(), DEFAULT_COLOR)

    def _draw_boxes(self, frame, fw: int, fh: int):
        with self.lock:
            preds = list(self.detections)

        for p in preds:
            x1 = max(0, min(p["x1"], fw - 1))
            y1 = max(0, min(p["y1"], fh - 1))
            x2 = max(0, min(p["x2"], fw - 1))
            y2 = max(0, min(p["y2"], fh - 1))
            color = self._color(p["class"])
            label = f"{p['class']}  {p['confidence']:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            py = max(y1 - 4, th + 6)
            cv2.rectangle(frame, (x1, py - th - bl - 3), (x1 + tw + 6, py + 3),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 3, py - bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def _draw_panel(self, frame):
        src_label = f"RTSP ({self.rtsp_transport.upper()})" if self.is_rtsp else "Webcam/File"
        conn_label = "🟢 LIVE" if self.stream_ok else "🔴 CONNECTING…"
        lines = [
            "  ROBOT VISION SYSTEM  ",
            f"  FPS: {self.fps:.1f}   Frame: {self.frame_count}",
            f"  Live: {len(self.detections)}   Total: {self.total_detections}",
            f"  Interval: {self.interval}s   Conf: {self.confidence:.0%}",
            f"  {src_label}  {conn_label}",
            "  [Q]uit [S]creen [D]etect [F]req",
        ]
        lh, pad = 19, 6
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (330, len(lines) * lh + pad * 2), (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

        for i, line in enumerate(lines):
            col = (0, 240, 255) if i == 0 else (240, 240, 240)
            cv2.putText(frame, line, (pad, pad + (i + 1) * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.49, col, 1, cv2.LINE_AA)
        if self.detecting:
            cv2.circle(frame, (318, 12), 7, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (318, 12), 7, (255, 255, 255), 1)
        return frame

    # ── Main loop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        print("🤖  Robot Vision System")
        print("=" * 55)
        print("Controls:  [Q] Quit  [S] Screenshot  [D] Detect  [F] Freq\n")

        if self.is_rtsp:
            print(f"📡  RTSP mode  ({self.rtsp_transport.upper()})")
            print(f"    Source : {self.cam_src}")
            print(f"    Auto-reconnects every {self.reconnect_delay}s on drop\n")
        else:
            print(f"🎥  Camera source: {self.cam_src}\n")

        win = "Robot Vision System"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        fw = fh = 0

        while self.running:
            # ── Open / re-open capture ────────────────────────────────────
            print(f"🔌  Connecting to {self.cam_src} …")
            cap = self._open_capture()

            if not cap.isOpened():
                self.reconnect_attempts += 1
                print(f"❌  Could not open stream  (attempt {self.reconnect_attempts})")
                if not self.is_rtsp:
                    print("    Check CAMERA_SOURCE in .env")
                    break
                print(f"    Retrying in {self.reconnect_delay}s…")
                time.sleep(self.reconnect_delay)
                continue

            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or fw
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fh
            print(f"✅  Connected  ({fw}×{fh})   attempt #{self.reconnect_attempts + 1}")
            self.reconnect_attempts = 0

            # Start dedicated frame-reader thread
            reader = threading.Thread(target=self._frame_reader, args=(cap,), daemon=True)
            reader.start()

            # ── Per-connection display loop ───────────────────────────────
            while self.running:
                try:
                    frame = self._frame_queue.get(timeout=5.0)
                except queue.Empty:
                    print("⚠️   No frame received for 5 s — reconnecting…")
                    break

                if frame is None:
                    print("⚠️   Stream dropped — reconnecting…")
                    break

                # Update frame dimensions if not yet known
                if fw == 0:
                    fh, fw = frame.shape[:2]

                self.frame_count += 1
                now = time.time()

                # Trigger background inference
                if not self.detecting and (now - self.last_detect) >= self.interval:
                    self.detecting   = True
                    self.last_detect = now
                    threading.Thread(
                        target=self._infer,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()

                frame = self._draw_boxes(frame, fw, fh)
                frame = self._draw_panel(frame)

                # FPS counter
                self._fps_n += 1
                if (now - self._fps_t) >= 1.0:
                    self.fps    = self._fps_n / (now - self._fps_t)
                    self._fps_t = now
                    self._fps_n = 0

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n👋  Shutting down…")
                    self.running = False
                elif key == ord("s"):
                    fn = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fn, frame)
                    print(f"📸  Saved: {fn}")
                elif key == ord("d"):
                    self.last_detect = 0.0
                elif key == ord("f"):
                    self.interval = 1.0 if self.interval == 0.5 else 0.5
                    print(f"⏱️   Interval → {self.interval}s")

                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False

            # ── Clean up this connection before reconnecting ──────────────
            cap.release()
            reader.join(timeout=2.0)
            self.stream_ok = False

            if self.running and self.is_rtsp:
                self.reconnect_attempts += 1
                print(f"    Reconnecting in {self.reconnect_delay}s…  "
                      f"(attempt {self.reconnect_attempts})")
                time.sleep(self.reconnect_delay)

        cv2.destroyAllWindows()
        print("✅  Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RobotVisionSystem().start()