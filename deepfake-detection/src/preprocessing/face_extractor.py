"""
face_extractor.py
------------------
Reads a video with OpenCV, samples frames, detects the face with MediaPipe
Face Mesh, and crops a square face region ready for the CNN.

MediaPipe is used instead of dlib because it ships its own model weights
(no separate 68-landmark .dat file to download) and gives us the eye
landmarks we need for blink detection for free.
"""
import cv2
import numpy as np
import mediapipe as mp

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for the left/right eye (MediaPipe's 468-point mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye_idx, image_w, image_h):
    """Compute the Eye Aspect Ratio (EAR) for blink detection.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Low EAR ~ eye closed, high EAR ~ eye open.
    """
    pts = np.array([
        (landmarks[i].x * image_w, landmarks[i].y * image_h) for i in eye_idx
    ])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


class FaceExtractor:
    def __init__(self, face_size=config.FACE_SIZE, min_detection_confidence=0.5):
        self.face_size = face_size
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame_bgr):
        """Detect the face in a single BGR frame.

        Returns (cropped_face_rgb, ear_value) or (None, None) if no face found.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Bounding box from all landmarks, with a bit of margin
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))

        margin_x = (x_max - x_min) * 0.25
        margin_y = (y_max - y_min) * 0.25
        x_min = int(max(0, x_min - margin_x))
        x_max = int(min(w, x_max + margin_x))
        y_min = int(max(0, y_min - margin_y))
        y_max = int(min(h, y_max + margin_y))

        face_crop = frame_bgr[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return None, None

        face_crop = cv2.resize(face_crop, (self.face_size, self.face_size))
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        return face_crop_rgb, avg_ear

    def extract_sequence(self, video_path, sequence_length=config.SEQUENCE_LENGTH,
                          stride=config.FRAME_SAMPLE_STRIDE):
        """Sample frames from a video, run face detection + EAR on each,
        and return a fixed-length sequence of face crops and their EAR values.

        Returns:
            faces: np.ndarray of shape (sequence_length, H, W, 3), dtype uint8
            ears:  np.ndarray of shape (sequence_length,), dtype float32
            success: bool, False if too few frames had a detectable face
        """
        cap = cv2.VideoCapture(video_path)
        faces, ears = [], []
        frame_idx = 0

        while cap.isOpened() and len(faces) < sequence_length * 3:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                face, ear = self.process_frame(frame)
                if face is not None:
                    faces.append(face)
                    ears.append(ear)
            frame_idx += 1

        cap.release()

        if len(faces) < max(2, sequence_length // 2):
            return None, None, False

        faces, ears = self._resample_to_length(faces, ears, sequence_length)
        return np.stack(faces), np.array(ears, dtype=np.float32), True

    @staticmethod
    def _resample_to_length(faces, ears, target_len):
        """Uniformly resample a variable-length list of frames to a fixed length
        (pad by repeating the last frame if too short, or evenly subsample if too long).
        """
        n = len(faces)
        if n == target_len:
            return faces, ears
        if n > target_len:
            idxs = np.linspace(0, n - 1, target_len).astype(int)
            return [faces[i] for i in idxs], [ears[i] for i in idxs]
        # n < target_len: pad with the last frame
        pad_needed = target_len - n
        return faces + [faces[-1]] * pad_needed, ears + [ears[-1]] * pad_needed

    def close(self):
        self.face_mesh.close()
