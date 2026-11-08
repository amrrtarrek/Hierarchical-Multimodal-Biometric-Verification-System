"""
src/live_face.py
Task 1.4 Real-Time Live Webcam Face Verification Stream.

Status logic (task sheet):
  genuine and P >= 0.85          -> GREEN  "VERIFIED: <Name> (Conf%)"
  genuine and 0.50 <= P < 0.85   -> YELLOW "AMBIGUOUS: Escalating"
  Class 0 or P < 0.50            -> RED    "ACCESS DENIED: Impostor"
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import joblib
from src.face_classifiers import FaceSVMModel, FaceRFModel
from src.face_extractors import HOGExtractor, LBPExtractor, PCAFaceExtractor

# Allow artifacts that were pickled while training ran as __main__.
import __main__
for _cls in (FaceSVMModel, FaceRFModel, HOGExtractor, LBPExtractor, PCAFaceExtractor):
    setattr(__main__, _cls.__name__, _cls)

from src.face_preprocessing import CASCADE_PATH, get_cascade
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face", "best_face_model.joblib")

CONFIDENCE_VERIFIED = 0.85
CONFIDENCE_AMBIGUOUS = 0.50
DETECT_EVERY_N_FRAMES = 3      # cascade runs every 3rd frame; boxes are re-drawn every frame
PADDING = 0.0                  # must match src/face_preprocessing.py (strict bounding-box crop)
LOG_EVERY_SECONDS = 1.0

GREEN, YELLOW, RED = (0, 255, 0), (0, 255, 255), (0, 0, 255)


def crop_and_preprocess(gray_frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop (same 25% padding as training), equalise and resize to 100x100."""
    img_h, img_w = gray_frame.shape[:2]
    pad_x, pad_y = int(w * PADDING), int(h * PADDING)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
    crop = gray_frame[y1:y2, x1:x2]
    return cv2.resize(cv2.equalizeHist(crop), (100, 100), interpolation=cv2.INTER_AREA)


def preprocess_crop(crop_gray: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.equalizeHist(crop_gray), (100, 100), interpolation=cv2.INTER_AREA)


def classify_status(pred_class: int, confidence: float) -> str:
    """Return 'VERIFIED', 'AMBIGUOUS' or 'DENIED' following the task-sheet rules."""
    if pred_class == 0 or confidence < CONFIDENCE_AMBIGUOUS:
        return "DENIED"
    if confidence >= CONFIDENCE_VERIFIED:
        return "VERIFIED"
    return "AMBIGUOUS"


def draw_result(frame, x, y, w, h, pred_class, name, confidence):
    pct = confidence * 100.0
    status = classify_status(pred_class, confidence)
    if status == "VERIFIED":
        color, label = GREEN, f"VERIFIED: {name} ({pct:.1f}%)"
    elif status == "AMBIGUOUS":
        color, label = YELLOW, f"AMBIGUOUS: Escalating ({pct:.1f}%)"
    else:
        color, label = RED, "ACCESS DENIED: Impostor"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    t_sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ly = y - 27 if y >= 27 else y + h + 3          # label strip above the box (below if no room)
    cv2.rectangle(frame, (x, ly), (x + t_sz[0] + 10, ly + 24), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 5, ly + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


class ProbaSmoother:
    """Tracks faces by box centre and averages their probability vectors over the last N detections."""

    def __init__(self, window: int = 8, max_dist: float = 80.0, max_miss: int = 5):
        self.window, self.max_dist, self.max_miss = window, max_dist, max_miss
        self.tracks = []   # each: {"c": (x, y), "buf": [proba, ...], "seen": bool, "miss": int}

    def update(self, center, proba):
        best, best_d = None, self.max_dist
        for t in self.tracks:
            d = float(np.hypot(center[0] - t["c"][0], center[1] - t["c"][1]))
            if d < best_d:
                best, best_d = t, d
        if best is None:
            best = {"c": center, "buf": [], "seen": True, "miss": 0}
            self.tracks.append(best)
        best["c"], best["seen"], best["miss"] = center, True, 0
        best["buf"] = (best["buf"] + [proba])[-self.window:]
        return np.mean(best["buf"], axis=0)

    def prune(self):
        for t in self.tracks:
            t["miss"] = 0 if t["seen"] else t["miss"] + 1
            t["seen"] = False
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]


def run_live_verification(model_path: str = MODEL_PATH, camera_id: int = 0):
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}\n        Run 'python src/face_classifiers.py' first.")
        return
    artifact = joblib.load(model_path)
    extractor, model, label_names = artifact["extractor"], artifact["model"], artifact["label_names"]
    print(f"[INFO] Loaded {artifact.get('best_config', 'model')} "
          f"(CV Macro F1={artifact.get('best_f1', float('nan')):.4f})")

    face_cascade = get_cascade()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    frame_count, fps_timer, fps_display = 0, time.time(), 0.0
    last_results, last_log = [], 0.0
    smoother = ProbaSmoother()
    print("[INFO] Live Face Verification active. Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            last_results = []
            for (x, y, w, h) in faces:
                proba = model.predict_proba(extractor.transform([crop_and_preprocess(gray, x, y, w, h)]))[0]
                proba = smoother.update((x + w / 2, y + h / 2), proba)   # average over recent detections
                idx = int(np.argmax(proba))
                p_class, conf = int(model.classes_[idx]), float(proba[idx])
                name = label_names.get(p_class, f"Subject_{p_class:03d}")
                last_results.append((x, y, w, h, p_class, name, conf))
                if time.time() - last_log >= LOG_EVERY_SECONDS:
                    print(f"[LIVE] {name} (Class {p_class}) | P={conf*100:.1f}% | {classify_status(p_class, conf)}")
                    last_log = time.time()
            smoother.prune()

        for (x, y, w, h, p_class, name, conf) in last_results:
            draw_result(frame, x, y, w, h, p_class, name, conf)

        if time.time() - fps_timer >= 1.0:
            fps_display = frame_count / (time.time() - fps_timer)
            frame_count, fps_timer = 0, time.time()
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Face Verification | Q=Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_verification()
