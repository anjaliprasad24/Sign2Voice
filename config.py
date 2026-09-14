"""
Configuration constants for the backend server.
Shared across MediaPipe processing, inference, and state machine logic.
"""

# --- Server ---
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# --- Video / Temporal Windowing ---
TARGET_FPS = 30
SEQUENCE_LENGTH = 30          # frames per prediction window (~1 second)
NUM_LANDMARKS = 21            # MediaPipe hand landmarks
NUM_ANGLE_FEATURES = 63       # matches ML contract: model input (1, 30, 63)

# --- MediaPipe Confidence / Speed ---
STATIC_IMAGE_MODE = False     # video-stream tracking, not per-frame detection
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE = 0.75

# --- Debounce / State Machine ---
CONFIDENCE_THRESHOLD = 0.85
MIN_CONSECUTIVE_FRAMES = 10
NEUTRAL_LABEL = "NEUTRAL"
PAUSE_DURATION_SEC = 1.5      # hand-drop pause before inserting a space

# --- CLAHE (Low-light handling) ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- Model artifacts (delivered by Teammate 1 - ML Lead) ---
MODEL_PATH = "model/gesture_model.keras"
LABELS_PATH = "model/labels.json"
