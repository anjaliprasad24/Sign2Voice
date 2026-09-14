"""
Sequence buffering + LSTM inference.

ML Contract with Teammate 1 (ML Lead):
    - model.predict() accepts input of shape (1, 30, 63)
    - output is a softmax probability array aligned with model/labels.json
    - model/normalize.py exposes:
          extract_features(hands_data) -> np.ndarray of shape (63,)
      which wrist-centers and scale-normalizes landmarks and computes the
      21-joint connection angles for a single frame.

Until Teammate 1 delivers gesture_model.keras / labels.json / normalize.py,
this module automatically falls back to a mock model (USE_MOCK_MODEL) so
the backend and frontend can be built and tested end-to-end independently,
per the contract-first workflow.
"""
import json
import os
from collections import deque

import numpy as np

from config import MODEL_PATH, LABELS_PATH, SEQUENCE_LENGTH, NUM_ANGLE_FEATURES

USE_MOCK_MODEL = not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH))

if not USE_MOCK_MODEL:
    from tensorflow.keras.models import load_model
    from model.normalize import extract_features  # Teammate 1's deliverable

    _model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        _labels = json.load(f)
else:
    _model = None
    _labels = {"0": "NEUTRAL", "1": "Hello", "2": "Thank You", "3": "No"}

    def extract_features(hands_data):
        """
        Mock feature extractor, used only while waiting on Teammate 1's
        real model/normalize.py. Flattens whatever raw landmarks are
        available into a fixed-length (63,) vector so shapes always line
        up with the real ML contract.
        """
        if not hands_data:
            return np.zeros(NUM_ANGLE_FEATURES, dtype=np.float32)
        flat = np.array(hands_data[0], dtype=np.float32).flatten()
        vec = np.zeros(NUM_ANGLE_FEATURES, dtype=np.float32)
        n = min(len(flat), NUM_ANGLE_FEATURES)
        vec[:n] = flat[:n]
        return vec


class SequenceBuffer:
    """
    Rolling window of the last SEQUENCE_LENGTH per-frame feature vectors
    for a single client session, matching the 30-frame (~1 second)
    temporal window used during training.
    """

    def __init__(self):
        self.frames = deque(maxlen=SEQUENCE_LENGTH)

    def push(self, feature_vector):
        self.frames.append(feature_vector)

    def is_ready(self):
        return len(self.frames) == SEQUENCE_LENGTH

    def as_array(self):
        """Shape (1, 30, 63) - ready to hand directly to model.predict()."""
        return np.expand_dims(np.array(self.frames, dtype=np.float32), axis=0)

    def clear(self):
        self.frames.clear()


def predict(sequence_array):
    """
    Run the LSTM (or mock) model on a (1, 30, 63) sequence.
    Returns (predicted_label: str, confidence: float).
    """
    if USE_MOCK_MODEL:
        probs = np.random.dirichlet(np.ones(len(_labels)), size=1)[0]
    else:
        probs = _model.predict(sequence_array, verbose=0)[0]

    best_idx = int(np.argmax(probs))
    label = _labels[str(best_idx)]
    confidence = float(probs[best_idx])
    return label, confidence
