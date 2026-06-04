import cv2
import numpy as np

from numpy.linalg import norm
from ai_edge_litert.interpreter import Interpreter

from config import MODEL_PATH


class MobileFaceNet:

    def __init__(self):

        self.interpreter = Interpreter(
            model_path=str(MODEL_PATH)
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_embedding(self, face_img):

        face_img = cv2.resize(face_img, (112, 112))

        face_img = cv2.cvtColor(
            face_img,
            cv2.COLOR_BGR2RGB
        )

        face_img = face_img.astype(np.float32)

        face_img = (face_img - 127.5) / 128.0

        face_img = np.expand_dims(face_img, axis=0)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            face_img
        )

        self.interpreter.invoke()

        embedding = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        embedding /= norm(embedding)

        return embedding