import cv2
from arcface_helper import ArcFaceHelper

print("test_arcface.py started")

img = cv2.imread("test_face.jpg")
print("Image loaded:", img is not None)

helper = ArcFaceHelper()
print("About to start helper")
helper.start()
print("Helper started")

emb = helper.get_embedding(img)
print("Embedding is None:", emb is None)
print("Shape:", None if emb is None else emb.shape)

helper.stop()
print("Helper stopped")
