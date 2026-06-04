import cv2
import numpy as np

from mobilefacenet import MobileFaceNet


model = MobileFaceNet()

dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)

embedding = model.get_embedding(dummy_face)

print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))
print("First 10 values:", embedding[:10])