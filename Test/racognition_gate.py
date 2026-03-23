import threading
import time
import cv2
import numpy as np
from face_db import identify_face

MATCH_THRESHOLD = 0.90

def fake_embedding_from_face(face_crop):
    small = cv2.resize(face_crop, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    vec = gray.flatten().astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


class RecognitionGateThread(threading.Thread):
    def __init__(self, face_thread, zoom_ui, threshold=MATCH_THRESHOLD):
        super().__init__(daemon=True)
        self.face_thread = face_thread
        self.zoom_ui = zoom_ui
        self.threshold = threshold
        self.running = True
        self.lock = threading.Lock()

        self.recognized = False
        self.name = None
        self.score = 0.0

    def stop(self):
        self.running = False

    def get_result(self):
        with self.lock:
            return {
                "recognized": self.recognized,
                "name": self.name,
                "score": self.score
            }

    def run(self):
        while self.running:
            if self.face_thread is None or not self.face_thread.is_alive():
                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Waiting for recognized patient...")
                time.sleep(0.5)
                continue

            result = self.face_thread.get_result()
            face_seen = result["face_seen"]
            bbox = result["bbox"]
            frame = result["frame"]

            if not face_seen or bbox is None or frame is None:
                with self.lock:
                    self.recognized = False
                    self.name = None
                    self.score = 0.0

                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Waiting for recognized patient...")
                time.sleep(0.2)
                continue

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Face detected, checking identity...")
                time.sleep(0.1)
                continue

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Face detected, checking identity...")
                time.sleep(0.1)
                continue

            if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Move closer for identification")
                time.sleep(0.1)
                continue

            emb = fake_embedding_from_face(face_crop)
            name, score = identify_face(emb, threshold=self.threshold)

            with self.lock:
                self.recognized = name is not None and score >= self.threshold
                self.name = name
                self.score = score

            if name is not None and score >= self.threshold:
                self.zoom_ui.set_face_detected(True)
                self.zoom_ui.set_status(f"Recognized: {name} ({score:.2f}) — tap to connect")
            else:
                self.zoom_ui.set_face_detected(False)
                self.zoom_ui.set_status("Unrecognized face — access locked")

            time.sleep(0.3)
