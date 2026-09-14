"""
MediaPipe hand-landmark extraction, tuned for real-time streaming:
    - static_image_mode=False -> treats input as video, uses lightweight
      tracking after the first detection instead of re-detecting every frame
    - min_detection/tracking_confidence raised to 0.75 so that when hands
      self-occlude, MediaPipe drops the unstable track and re-invokes the
      palm detector instead of feeding guessed coordinates downstream
    - frame.flags.writeable = False to pass frames by reference, saving
      a memory copy on every single frame
"""
import base64

import cv2
import numpy as np
import mediapipe as mp

from config import (
    STATIC_IMAGE_MODE,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

mp_hands = mp.solutions.hands

_hands = mp_hands.Hands(
    static_image_mode=STATIC_IMAGE_MODE,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)


def decode_base64_frame(base64_str):
    """
    Decode a base64-encoded JPEG/PNG string (sent from the frontend's
    hidden <canvas> over the WebSocket) into an OpenCV BGR numpy array.
    Returns None if decoding fails so the caller can skip the frame.
    """
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def extract_landmarks(frame_bgr):
    """
    Run MediaPipe hand tracking on a single frame.

    Returns:
        list[list[tuple(x, y, z)]] - one list of 21 (x, y, z) landmarks
        per detected hand (up to MAX_NUM_HANDS), or [] if no hand was
        found in this frame.
    """
    frame_bgr.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _hands.process(frame_rgb)
    frame_bgr.flags.writeable = True

    if not results.multi_hand_landmarks:
        return []

    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        hands_data.append(coords)
    return hands_data


def close():
    """Release MediaPipe resources on server shutdown."""
    _hands.close()
