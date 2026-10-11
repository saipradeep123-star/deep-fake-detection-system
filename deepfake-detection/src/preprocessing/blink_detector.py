"""
blink_detector.py
------------------
Turns a per-frame Eye Aspect Ratio (EAR) sequence into blink-pattern
features. Early deepfake generators struggled to reproduce natural
blink rates, so blink dynamics are a useful auxiliary signal alongside
the CNN's spatial features.
"""
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def count_blinks(ear_sequence, threshold=config.EAR_THRESHOLD,
                  consec_frames=config.EAR_CONSEC_FRAMES):
    """Count blinks in an EAR sequence using a simple threshold + run-length rule.

    A blink is counted when EAR drops below `threshold` for at least
    `consec_frames` consecutive frames and then recovers.
    """
    below = ear_sequence < threshold
    blinks = 0
    run_length = 0

    for is_closed in below:
        if is_closed:
            run_length += 1
        else:
            if run_length >= consec_frames:
                blinks += 1
            run_length = 0
    if run_length >= consec_frames:
        blinks += 1

    return blinks


def blink_feature_vector(ear_sequence):
    """Summarize an EAR sequence into a small fixed-size feature vector:
    [blink_count, mean_ear, std_ear, min_ear, max_ear].

    This can be concatenated with the LSTM's final hidden state before
    the classification head for a lightweight fusion of handcrafted +
    learned temporal features.
    """
    ear_sequence = np.asarray(ear_sequence, dtype=np.float32)
    return np.array([
        count_blinks(ear_sequence),
        float(np.mean(ear_sequence)),
        float(np.std(ear_sequence)),
        float(np.min(ear_sequence)),
        float(np.max(ear_sequence)),
    ], dtype=np.float32)
