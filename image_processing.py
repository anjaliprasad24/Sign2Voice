"""
Low-light / harsh-light preprocessing using CLAHE (Contrast Limited
Adaptive Histogram Equalization). Applied to every frame before it
reaches MediaPipe, since poor contrast degrades edge-based joint
detection before coordinates ever reach the LSTM.
"""
import cv2

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)


def apply_clahe(frame_bgr):
    """
    Convert frame to LAB color space and apply CLAHE to the L (luminance)
    channel only, then convert back to BGR. Operating on luminance alone
    (rather than each RGB channel) lifts shadow detail and balances
    blown-out highlights without distorting hue/color.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = _clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def estimate_brightness(frame_bgr):
    """
    Rough 0-255 brightness estimate, useful for backend-side logging when
    debugging tracking loss. The authoritative lighting guard shown to the
    user lives in the frontend (Teammate 3).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.mean()
