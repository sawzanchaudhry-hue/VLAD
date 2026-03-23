#!/usr/bin/env python3

import cv2
import numpy as np

MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"

recognizer = cv2.FaceRecognizerSF_create(MODEL_PATH, "")


def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def get_face_embedding(face_crop):
    """
    Temporary real embedding function using direct face crop.
    Later we can improve this with landmark-based alignment.
    """
    if face_crop is None or face_crop.size == 0:
        return None

    try:
        resized = cv2.resize(face_crop, (112, 112))
        feat = recognizer.feature(resized)
        feat = np.asarray(feat, dtype=np.float32).flatten()
        return l2_normalize(feat)
    except Exception as e:
        print("Embedding error:", e)
        return None
