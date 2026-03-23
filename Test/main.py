#!/usr/bin/env python3

import time
import cv2
import numpy as np
from faced import FaceDetectionThread
from face_db import identify_face
from face_recognizer import get_face_embedding

MATCH_THRESHOLD = 0.75
MATCH_INTERVAL = 1.0  # seconds




def main():
    print("Starting face recognition test")

    face_thread = FaceDetectionThread(show_window=False)
    face_thread.start()

    last_match_time = 0

    try:
        while True:
            result = face_thread.get_result()

            face_seen = result["face_seen"]
            bbox = result["bbox"]
            frame = result["frame"]

            if not face_seen or bbox is None or frame is None:
                print("No face detected")
                time.sleep(0.5)
                continue

            now = time.time()
            if now - last_match_time < MATCH_INTERVAL:
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
                print("Bad bbox")
                time.sleep(0.1)
                continue

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                print("Empty crop")
                time.sleep(0.1)
                continue

            if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                print("Face too small")
                time.sleep(0.1)
                continue

            emb = get_face_embedding(face_crop)
            if emb is None:
                print("Failed to generate embedding")
                time.sleep(0.1)
                continue
            name, score = identify_face(emb, threshold=MATCH_THRESHOLD)

            if name is None:
                print(f"Unknown face (score={score:.3f})")
            else:
                print(f"Recognized: {name} (score={score:.3f})")

            last_match_time = now

    except KeyboardInterrupt:
        print("\nStopping recognition test")

    finally:
        face_thread.stop()
        face_thread.join()
        print("Face thread fully stopped")


if __name__ == "__main__":
    main()
