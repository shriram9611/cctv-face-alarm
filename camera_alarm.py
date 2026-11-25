"""Laptop-only face recognition alarm. No microphone or audio classification."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PEOPLE_FILE = DATA_DIR / "known_faces.json"
ALARM_FILE = DATA_DIR / "alarm.wav"
DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DEFAULT_THRESHOLD = 0.363  # OpenCV Zoo SFace cosine-similarity example threshold
CONFIRM_FRAMES = 3
ALARM_COOLDOWN_SECONDS = 8


def ensure_models() -> None:
    missing = [path.name for path in (DETECTOR_MODEL, RECOGNIZER_MODEL) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Face models are missing: " + ", ".join(missing)
            + ". Run `python setup_models.py` once while online."
        )


def load_models():
    ensure_models()
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_MODEL), "", (320, 320), 0.85, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")
    return detector, recognizer


def detect_faces(frame, detector):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    return [] if faces is None else faces


def embedding_for(frame, face, recognizer) -> np.ndarray:
    aligned = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned).reshape(-1).astype(np.float32)


def load_people() -> dict[str, list[list[float]]]:
    if not PEOPLE_FILE.exists():
        return {}
    try:
        data = json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))
        return {
            str(name): [list(map(float, vector)) for vector in vectors]
            for name, vectors in data.items()
        }
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not read {PEOPLE_FILE}: {exc}") from exc


def save_people(people: dict[str, list[list[float]]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = PEOPLE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(people, indent=2), encoding="utf-8")
    temporary.replace(PEOPLE_FILE)


def match_person(embedding: np.ndarray, people: dict[str, list[list[float]]], threshold: float):
    best_name = "Unknown"
    best_score = -1.0
    for name, vectors in people.items():
        for reference in vectors:
            ref = np.asarray(reference, dtype=np.float32)
            score = float(np.dot(embedding, ref) / (
                (np.linalg.norm(embedding) * np.linalg.norm(ref)) + 1e-12
            ))
            if score > best_score:
                best_name, best_score = name, score
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score


def create_alarm_file() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ALARM_FILE.exists():
        sample_rate = 24000
        duration = 0.45
        with wave.open(str(ALARM_FILE), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for index in range(int(sample_rate * duration)):
                time_value = index / sample_rate
                frequency = 880 if int(time_value * 8) % 2 == 0 else 660
                envelope = min(1.0, index / 200, (sample_rate * duration - index) / 200)
                value = int(9000 * envelope * math.sin(2 * math.pi * frequency * time_value))
                wav.writeframesraw(struct.pack("<h", value))
    return ALARM_FILE


class Alarm:
    def __init__(self):
        self.path = create_alarm_file()
        self.process = None
        system = platform.system()
        if system == "Darwin":
            self.command = ["/usr/bin/afplay", str(self.path)]
        elif system == "Windows":
            self.command = None
        else:
            player = shutil.which("paplay") or shutil.which("aplay") or shutil.which("ffplay")
            self.command = [player, "-nodisp", "-autoexit", str(self.path)] if player else None

    def play(self):
        if self.command:
            try:
                subprocess.Popen(self.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError as exc:
                print(f"Could not start system audio player: {exc}")
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.PlaySound(str(self.path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except RuntimeError as exc:
                print(f"Could not play alert: {exc}")
        print("No supported audio player found; unknown face detected but alarm did not play.")


def open_camera(index: int):
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {index}. Check camera permission and close other camera apps."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def draw_face(frame, face, label, score):
    x, y, width, height = map(int, face[:4])
    color = (0, 180, 0) if label != "Unknown" else (0, 0, 230)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    text = f"{label} ({score:.2f})" if score >= 0 else label
    cv2.putText(frame, text, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def enroll(name: str, camera_index: int) -> None:
    if not name.strip():
        raise RuntimeError("Enter a non-empty person name.")
    detector, recognizer = load_models()
    camera = open_camera(camera_index)
    print("Enrollment: face the camera. Press S to save one face, or Q to cancel.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            faces = detect_faces(frame, detector)
            preview = frame.copy()
            for face in faces:
                draw_face(preview, face, "Face detected", -1)
            cv2.putText(preview, "S: save  |  Q: cancel", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Laptop Face Alarm - Enrollment", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Enrollment cancelled.")
                return
            if key == ord("s"):
                if len(faces) != 1:
                    print(f"Show exactly one face before saving (detected {len(faces)}).")
                    continue
                people = load_people()
                people.setdefault(name.strip(), []).append(embedding_for(frame, faces[0], recognizer).tolist())
                save_people(people)
                print(f"Saved one local face reference for {name.strip()}.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()


def scan(camera_index: int, threshold: float) -> None:
    people = load_people()
    if not people:
        raise RuntimeError(
            "No known people enrolled yet. Run `python camera_alarm.py enroll \"Name\"` first."
        )
    detector, recognizer = load_models()
    camera = open_camera(camera_index)
    alarm = Alarm()
    unknown_streak = 0
    last_alarm = 0.0
    print("Scanning locally. Press Q in the preview to stop. No microphone is opened.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            faces = detect_faces(frame, detector)
            unknown_found = False
            for face in faces:
                name, score = match_person(embedding_for(frame, face, recognizer), people, threshold)
                draw_face(frame, face, name, score)
                unknown_found = unknown_found or name == "Unknown"
            unknown_streak = unknown_streak + 1 if unknown_found else 0
            if unknown_streak >= CONFIRM_FRAMES:
                cv2.putText(frame, "UNKNOWN PERSON - ALARM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                now = time.monotonic()
                if now - last_alarm >= ALARM_COOLDOWN_SECONDS:
                    print("Unknown face confirmed. Playing alarm through laptop speakers.")
                    alarm.play()
                    last_alarm = now
            else:
                cv2.putText(frame, "Monitoring", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Laptop Face Alarm - Scan", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Local laptop-camera face recognition with a speaker alarm.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    enroll_parser = subparsers.add_parser("enroll", help="Register a consenting known person")
    enroll_parser.add_argument("name", help="Label to show for this known person")
    enroll_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    scan_parser = subparsers.add_parser("scan", help="Recognize faces and alarm on unknown people")
    scan_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    scan_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                             help="Cosine similarity threshold (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "enroll":
            enroll(args.name, args.camera)
        else:
            if not 0.0 <= args.threshold <= 1.0:
                raise RuntimeError("Threshold must be between 0 and 1.")
            scan(args.camera, args.threshold)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
