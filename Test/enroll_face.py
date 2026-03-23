#!/usr/bin/env python3

import time
import cv2
import numpy as np
from faced import FaceDetectionThread
from face_db import save_identity


NAME = "Ibrahim"
NUM_SAMPLES = 15
CAPTURE_DELAY = 1.0   # seconds between samples


from face_recognizer import get_face_embedding


def main():
    print("Starting face enrollment for:", NAME)

    face_thread = FaceDetectionThread(show_window=False)
    face_thread.start()

    embeddings = []
    last_capture_time = 0

    try:
        while len(embeddings) < NUM_SAMPLES:
            result = face_thread.get_result()

            face_seen = result["face_seen"]
            bbox = result["bbox"]
            frame = result["frame"]

            if not face_seen or bbox is None or frame is None:
                print("Waiting for face...")
                time.sleep(0.2)
                continue

            now = time.time()
            if now - last_capture_time < CAPTURE_DELAY:
                time.sleep(0.05)
                continue

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            box_w = x2 - x1
            box_h = y2 - y1

            pad_x = int(box_w * 0.15)
            pad_y = int(box_h * 0.20)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                print("Bad bbox, skipping")
                time.sleep(0.1)
                continue

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                print("Empty crop, skipping")
                time.sleep(0.1)
                continue

            # Minimum size check
            if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                print("Face too small, move closer")
                time.sleep(0.1)
                continue

            emb = get_face_embedding(face_crop)
            if emb is None:
                print("Failed to generate embedding")
                time.sleep(0.1)
                continue
            embeddings.append(emb)
            last_capture_time = now

            print(f"Captured sample {len(embeddings)}/{NUM_SAMPLES}")

        avg_embedding = np.mean(np.stack(embeddings), axis=0)
        save_identity(NAME, avg_embedding)

        print(f"Enrollment complete for {NAME}")
        print("Saved to face_db.json")

    finally:
        face_thread.stop()
        face_thread.join()
        print("Face thread fully stopped")


if __name__ == "__main__":
    main()
