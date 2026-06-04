import time
import random
from enum import Enum
from collections import deque

import cv2
import mediapipe as mp

from config import (
    APP_NAME,
    WINDOW_NAME,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MATCH_THRESHOLD,
    MATCH_MARGIN,
    TARGET_SAMPLES_PER_STAGE,
    ENROLLMENT_STAGES,
    SUCCESS_COLOR,
    WARNING_COLOR,
    ERROR_COLOR,
)

from mobilefacenet import MobileFaceNet
from database import (
    create_profile,
    save_database,
    load_database,
    match_database,
)
from liveness import extract_liveness_features
from utils import draw_text
from sync_manager import add_auth_event, mock_sync_to_aws


class Mode(Enum):
    IDLE = "IDLE"
    ENROLLING = "ENROLLING"
    READY = "READY"


class Challenge(Enum):
    BLINK = "BLINK"
    SMILE = "SMILE"
    TURN_LEFT = "TURN LEFT"
    TURN_RIGHT = "TURN RIGHT"
    CENTER = "CENTER FACE"
    COMPLETE = "LIVE VERIFIED"


def create_random_challenge():
    challenges = [
        Challenge.BLINK,
        Challenge.SMILE,
        Challenge.TURN_LEFT,
        Challenge.TURN_RIGHT,
    ]

    selected = random.sample(challenges, 3)
    selected.append(Challenge.CENTER)

    return selected


def bbox_from_detection(detection, width, height):
    bbox = detection.location_data.relative_bounding_box

    x1 = int(bbox.xmin * width)
    y1 = int(bbox.ymin * height)
    x2 = int((bbox.xmin + bbox.width) * width)
    y2 = int((bbox.ymin + bbox.height) * height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * 0.25)
    pad_y = int(box_height * 0.30)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return x1, y1, x2, y2


def reset_challenge():
    sequence = create_random_challenge()
    return sequence, 0, sequence[0]


def main():
    print(f"\n{APP_NAME}")
    print("Press E to enroll")
    print("Press L to list profiles")
    print("Press R to reset liveness")
    print("Press S to sync pending logs")
    print("Press D to delete local profiles")
    print("Press Q to quit\n")

    embedder = MobileFaceNet()

    profiles = load_database()
    mode = Mode.READY if profiles else Mode.IDLE

    print(f"Loaded profiles: {len(profiles)}")

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    current_name = None
    stage_embeddings = []
    all_enrollment_embeddings = []
    current_stage_index = 0

    challenge_sequence, challenge_index, challenge = reset_challenge()
    blink_frames = 0

    raw_score = 0.0
    second_score = 0.0
    last_score = 0.0
    last_match_name = "UNKNOWN"

    stable_name = "UNKNOWN"
    candidate_name = "UNKNOWN"
    candidate_count = 0
    REQUIRED_STABLE_FRAMES = 8

    last_auth_logged = False
    score_history = deque(maxlen=30)

    last_inference_ms = 0.0
    fps = 0.0
    last_time = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=25)

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        elapsed = max(now - last_time, 1e-6)
        fps = (fps * 0.9) + ((1.0 / elapsed) * 0.1)
        last_time = now

        features = None
        mesh_results = face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            features = extract_liveness_features(
                face_landmarks,
                width,
                height,
            )

            challenge_done = False

            if challenge == Challenge.BLINK:
                if features["blink"]:
                    blink_frames += 1
                else:
                    if blink_frames >= 2:
                        challenge_done = True
                    blink_frames = 0

            elif challenge == Challenge.SMILE:
                if features["smile"]:
                    challenge_done = True

            elif challenge == Challenge.TURN_LEFT:
                if features["pose"] == "LEFT":
                    challenge_done = True

            elif challenge == Challenge.TURN_RIGHT:
                if features["pose"] == "RIGHT":
                    challenge_done = True

            elif challenge == Challenge.CENTER:
                if features["pose"] == "FRONT":
                    challenge_done = True

            if challenge_done:
                challenge_index += 1
                blink_frames = 0

                if challenge_index >= len(challenge_sequence):
                    challenge = Challenge.COMPLETE
                else:
                    challenge = challenge_sequence[challenge_index]

        detection_results = face_detector.process(rgb_frame)

        face_box = None
        embedding = None

        if detection_results.detections:
            face_box = bbox_from_detection(
                detection_results.detections[0],
                width,
                height,
            )

            x1, y1, x2, y2 = face_box
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                start = time.perf_counter()
                embedding = embedder.get_embedding(face_crop)
                last_inference_ms = (time.perf_counter() - start) * 1000.0

        label = "NO FACE"
        color = ERROR_COLOR

        if face_box is not None:
            x1, y1, x2, y2 = face_box

            if embedding is not None:
                if mode == Mode.ENROLLING:
                    stage = ENROLLMENT_STAGES[current_stage_index]

                    if features is None:
                        pose_ok = False
                    elif stage == "FRONT":
                        pose_ok = features["pose"] == "FRONT"
                    elif stage == "SMILE":
                        pose_ok = features["pose"] == "FRONT" and features["smile"]
                    elif stage == "LEFT":
                        pose_ok = features["pose"] == "LEFT"
                    elif stage == "RIGHT":
                        pose_ok = features["pose"] == "RIGHT"
                    elif stage == "UP":
                        pose_ok = features["pose"] == "UP"
                    elif stage == "DOWN":
                        pose_ok = features["pose"] == "DOWN"
                    else:
                        pose_ok = False

                    if pose_ok:
                        stage_embeddings.append(embedding)
                        all_enrollment_embeddings.append(embedding)

                        label = (
                            f"ENROLL {current_name}: {stage} "
                            f"{len(stage_embeddings)}/{TARGET_SAMPLES_PER_STAGE}"
                        )
                        color = SUCCESS_COLOR

                        if len(stage_embeddings) >= TARGET_SAMPLES_PER_STAGE:
                            print(
                                f"Stage complete: {stage} "
                                f"({len(stage_embeddings)} samples)"
                            )

                            current_stage_index += 1
                            stage_embeddings = []

                            if current_stage_index >= len(ENROLLMENT_STAGES):
                                print(
                                    "Total samples saved:",
                                    len(all_enrollment_embeddings),
                                )

                                profile = create_profile(
                                    current_name,
                                    all_enrollment_embeddings,
                                )

                                if profile is not None:
                                    profiles.append(profile)
                                    save_database(profiles)
                                    print(f"Enrollment complete: {current_name}")

                                current_name = None
                                stage_embeddings = []
                                all_enrollment_embeddings = []
                                current_stage_index = 0

                                mode = Mode.READY
                                challenge_sequence, challenge_index, challenge = reset_challenge()
                                blink_frames = 0
                                score_history.clear()
                                stable_name = "UNKNOWN"
                                candidate_name = "UNKNOWN"
                                candidate_count = 0
                                last_match_name = "UNKNOWN"
                                last_auth_logged = False

                            else:
                                print(
                                    f"Next stage: "
                                    f"{ENROLLMENT_STAGES[current_stage_index]}"
                                )
                    else:
                        label = f"NEED {stage}"
                        color = WARNING_COLOR

                elif mode == Mode.READY and profiles:
                    best_profile, raw_score, second_score = match_database(
                        profiles,
                        embedding,
                    )

                    margin = raw_score - second_score

                    score_history.append(raw_score)
                    last_score = sum(score_history) / len(score_history)

                    if best_profile is not None:
                        possible_name = best_profile["name"]
                    else:
                        possible_name = "UNKNOWN"

                    match_ok = (
                        raw_score >= MATCH_THRESHOLD
                        and margin >= MATCH_MARGIN
                    )

                    if not match_ok:
                        possible_name = "UNKNOWN"

                    if possible_name == candidate_name:
                        candidate_count += 1
                    else:
                        candidate_name = possible_name
                        candidate_count = 1

                    if candidate_count >= REQUIRED_STABLE_FRAMES:
                        stable_name = candidate_name

                    last_match_name = stable_name

                    live_ok = challenge == Challenge.COMPLETE
                    auth_ok = live_ok and stable_name != "UNKNOWN"

                    if auth_ok:
                        if not last_auth_logged:
                            add_auth_event(
                                stable_name,
                                raw_score,
                                "AUTH_OK",
                            )
                            last_auth_logged = True

                        label = f"AUTH OK {stable_name} {raw_score * 100:.1f}%"
                        color = SUCCESS_COLOR

                    elif stable_name != "UNKNOWN":
                        last_auth_logged = False
                        label = (
                            f"MATCH {stable_name} "
                            f"{raw_score * 100:.1f}% - DO LIVENESS"
                        )
                        color = WARNING_COLOR

                    else:
                        last_auth_logged = False
                        label = (
                            f"UNKNOWN FACE {raw_score * 100:.1f}% "
                            f"MARGIN {margin * 100:.1f}%"
                        )
                        color = ERROR_COLOR

                else:
                    label = "PRESS E TO ENROLL"
                    color = WARNING_COLOR

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        draw_text(frame, f"MODE: {mode.value}  PROFILES: {len(profiles)}", 30)
        draw_text(frame, f"FPS: {fps:.1f}  INFER: {last_inference_ms:.0f} ms", 60)

        if features:
            if challenge == Challenge.COMPLETE:
                live_text = "LIVE VERIFIED"
            else:
                live_text = (
                    f"{challenge.value} "
                    f"({challenge_index + 1}/{len(challenge_sequence)})"
                )

            draw_text(
                frame,
                f"POSE: {features['pose']}  "
                f"EAR: {features['ear']:.2f}  "
                f"LIVENESS: {live_text}",
                90,
            )
        else:
            draw_text(frame, "NO LIVENESS DETECTED", 90)

        draw_text(
            frame,
            f"BEST: {last_match_name}  "
            f"SCORE: {raw_score * 100:.1f}%  "
            f"SECOND: {second_score * 100:.1f}%  "
            f"MARGIN: {(raw_score - second_score) * 100:.1f}%",
            120,
        )

        draw_text(
            frame,
            f"USER: {last_match_name}",
            150,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("e") and mode != Mode.ENROLLING:
            current_name = input("Enter name: ").strip()

            if not current_name:
                current_name = f"Person {len(profiles) + 1}"

            stage_embeddings = []
            all_enrollment_embeddings = []
            current_stage_index = 0

            stable_name = "UNKNOWN"
            candidate_name = "UNKNOWN"
            candidate_count = 0
            last_match_name = "UNKNOWN"
            last_auth_logged = False

            mode = Mode.ENROLLING
            challenge_sequence, challenge_index, challenge = reset_challenge()
            blink_frames = 0

            print(f"Enrollment started for {current_name}")
            print(f"Stage: {ENROLLMENT_STAGES[current_stage_index]}")

        elif key == ord("l"):
            print("\nProfiles:")
            for index, profile in enumerate(profiles, start=1):
                print(f"{index}. {profile['name']}")
            print()

        elif key == ord("r"):
            score_history.clear()
            challenge_sequence, challenge_index, challenge = reset_challenge()
            blink_frames = 0
            raw_score = 0.0
            second_score = 0.0
            last_score = 0.0
            stable_name = "UNKNOWN"
            candidate_name = "UNKNOWN"
            candidate_count = 0
            last_match_name = "UNKNOWN"
            last_auth_logged = False
            print("Session reset.")

        elif key == ord("s"):
            mock_sync_to_aws()

        elif key == ord("d"):
            profiles = []
            save_database(profiles)
            mode = Mode.IDLE
            challenge_sequence, challenge_index, challenge = reset_challenge()
            blink_frames = 0
            raw_score = 0.0
            second_score = 0.0
            last_score = 0.0
            stable_name = "UNKNOWN"
            candidate_name = "UNKNOWN"
            candidate_count = 0
            last_match_name = "UNKNOWN"
            last_auth_logged = False
            print("All local profiles deleted.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")