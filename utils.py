import cv2

from config import TEXT_COLOR, TEXT_BG


def draw_text(frame, text, y, color=TEXT_COLOR, scale=0.65):
    x = 18

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        2,
    )

    cv2.rectangle(
        frame,
        (x - 6, y - text_height - 7),
        (x + text_width + 8, y + baseline + 6),
        TEXT_BG,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )