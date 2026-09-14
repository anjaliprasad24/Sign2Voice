"""
Debounce / threshold state machine.

A prediction is only "committed" (emitted to the frontend) once it:
    1. exceeds CONFIDENCE_THRESHOLD, AND
    2. has been predicted consistently for MIN_CONSECUTIVE_FRAMES in a row.

The NEUTRAL/REST label acts as a delimiter: once it has been held for
PAUSE_DURATION_SEC, a space is inserted into the running sentence buffer
and the system is ready to commit the next word.
"""
import time

from config import (
    CONFIDENCE_THRESHOLD,
    MIN_CONSECUTIVE_FRAMES,
    NEUTRAL_LABEL,
    PAUSE_DURATION_SEC,
)


class GestureStateMachine:
    def __init__(self):
        self.current_label = None
        self.consecutive_count = 0
        self.last_committed_label = None
        self.neutral_start_time = None
        self.sentence_buffer = ""

    def update(self, label, confidence):
        """
        Feed in the latest per-window prediction.

        Returns:
            {
              "event": "none" | "word_committed" | "space_inserted",
              "label": str | None,
              "confidence": float | None,
              "sentence": str,
            }
        """
        result = {
            "event": "none",
            "label": None,
            "confidence": None,
            "sentence": self.sentence_buffer,
        }

        # Track consecutive identical predictions above threshold.
        if confidence >= CONFIDENCE_THRESHOLD and label == self.current_label:
            self.consecutive_count += 1
        elif confidence >= CONFIDENCE_THRESHOLD:
            self.current_label = label
            self.consecutive_count = 1
        else:
            self.current_label = None
            self.consecutive_count = 0

        # NEUTRAL acts as a delimiter/pause rather than a committed word.
        if self.current_label == NEUTRAL_LABEL:
            if self.neutral_start_time is None:
                self.neutral_start_time = time.time()
            elif (time.time() - self.neutral_start_time) >= PAUSE_DURATION_SEC:
                if self.sentence_buffer and not self.sentence_buffer.endswith(" "):
                    self.sentence_buffer += " "
                    result["event"] = "space_inserted"
                    result["sentence"] = self.sentence_buffer
                self.last_committed_label = None
            return result
        else:
            self.neutral_start_time = None

        # Commit a real word once it clears the debounce threshold.
        if (
            self.consecutive_count >= MIN_CONSECUTIVE_FRAMES
            and self.current_label != self.last_committed_label
        ):
            self.sentence_buffer += self.current_label
            self.last_committed_label = self.current_label
            result["event"] = "word_committed"
            result["label"] = self.current_label
            result["confidence"] = confidence
            result["sentence"] = self.sentence_buffer

        return result

    def reset_sentence(self):
        self.sentence_buffer = ""
        self.last_committed_label = None
        self.current_label = None
        self.consecutive_count = 0
        self.neutral_start_time = None
