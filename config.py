from pathlib import Path
DATABASE_PATH = Path("identities.enc")
SECRET_KEY_PATH = Path("secret.key")
SYNC_QUEUE_PATH = Path("sync_queue.enc")

APP_NAME = "LiteSecure Clean"
WINDOW_NAME = "LiteSecure - Offline Face Authentication"

MODEL_PATH = Path("models/mobilefacenet.tflite")
DATABASE_PATH = Path("identities.npz")

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MATCH_THRESHOLD = 0.60
MATCH_MARGIN = 0.09


ENROLLMENT_STAGES = [
    "FRONT",
    "SMILE",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
]

TARGET_SAMPLES_PER_STAGE = 150

TEXT_COLOR = (0, 0, 0)
TEXT_BG = (210, 210, 210)

SUCCESS_COLOR = (0, 170, 0)
WARNING_COLOR = (0, 160, 255)
ERROR_COLOR = (0, 0, 220)

BLINK_THRESHOLD = 0.20

LEFT_RIGHT_THRESHOLD = 0.16
UP_DOWN_THRESHOLD = 0.10

SMILE_THRESHOLD = 9.0
