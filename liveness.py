import numpy as np

from scipy.spatial import distance

from config import (
    BLINK_THRESHOLD,
    LEFT_RIGHT_THRESHOLD,
    UP_DOWN_THRESHOLD,
    SMILE_THRESHOLD,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14


def landmark_point(face_landmarks, index, width, height):
    landmark = face_landmarks.landmark[index]
    return np.array(
        [landmark.x * width, landmark.y * height],
        dtype=np.float32,
    )


def calculate_ear(points):
    vertical_1 = distance.euclidean(points[1], points[5])
    vertical_2 = distance.euclidean(points[2], points[4])
    horizontal = max(distance.euclidean(points[0], points[3]), 1.0)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def extract_liveness_features(face_landmarks, width, height):
    left_eye = [
        landmark_point(face_landmarks, idx, width, height)
        for idx in LEFT_EYE
    ]

    right_eye = [
        landmark_point(face_landmarks, idx, width, height)
        for idx in RIGHT_EYE
    ]

    ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

    nose = landmark_point(face_landmarks, NOSE_TIP, width, height)
    left_face = landmark_point(face_landmarks, LEFT_FACE, width, height)
    right_face = landmark_point(face_landmarks, RIGHT_FACE, width, height)

    forehead = landmark_point(face_landmarks, 10, width, height)
    chin = landmark_point(face_landmarks, 152, width, height)

    face_width = max(distance.euclidean(left_face, right_face), 1.0)
    face_height = max(distance.euclidean(forehead, chin), 1.0)

    face_center_x = (left_face[0] + right_face[0]) / 2.0
    face_center_y = (forehead[1] + chin[1]) / 2.0

    head_offset = (nose[0] - face_center_x) / face_width
    head_offset_y = (nose[1] - face_center_y) / face_height

    # Smile detection
    mouth_left = landmark_point(face_landmarks, MOUTH_LEFT, width, height)
    mouth_right = landmark_point(face_landmarks, MOUTH_RIGHT, width, height)
    upper_lip = landmark_point(face_landmarks, UPPER_LIP, width, height)
    lower_lip = landmark_point(face_landmarks, LOWER_LIP, width, height)

    mouth_width = distance.euclidean(mouth_left, mouth_right)
    mouth_open = max(distance.euclidean(upper_lip, lower_lip), 1.0)

    smile_ratio = mouth_width / mouth_open
    smile = smile_ratio > SMILE_THRESHOLD

    if head_offset < -LEFT_RIGHT_THRESHOLD:
        pose = "LEFT"
    elif head_offset > LEFT_RIGHT_THRESHOLD:
        pose = "RIGHT"
    elif head_offset_y < -UP_DOWN_THRESHOLD:
        pose = "UP"
    elif head_offset_y > UP_DOWN_THRESHOLD:
        pose = "DOWN"
    else:
        pose = "FRONT"

    blink = ear < BLINK_THRESHOLD

    return {
        "ear": ear,
        "head_offset": head_offset,
        "head_offset_y": head_offset_y,
        "pose": pose,
        "blink": blink,
        "smile_ratio": smile_ratio,
        "smile": smile,
    }
