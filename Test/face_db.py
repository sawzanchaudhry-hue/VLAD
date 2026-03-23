import json
import numpy as np
from pathlib import Path

DB_FILE = Path("face_db.json")

def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm

def cosine_similarity(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def save_identity(name, embedding):
    embedding = l2_normalize(embedding).tolist()
    data = {}
    if DB_FILE.exists():
        data = json.loads(DB_FILE.read_text())
    data[name] = embedding
    DB_FILE.write_text(json.dumps(data, indent=2))

def load_database():
    if not DB_FILE.exists():
        return {}
    raw = json.loads(DB_FILE.read_text())
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}

def identify_face(embedding, threshold=0.45):
    db = load_database()
    if not db:
        return None, 0.0

    best_name = None
    best_score = -1.0

    for name, ref_vec in db.items():
        score = cosine_similarity(embedding, ref_vec)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score
