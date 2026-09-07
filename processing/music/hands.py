"""MediaPipe hand landmarks, reduced to fingertips in image coordinates.

MediaPipe is an optional extra; imports stay lazy so the rest of the package
works without it. Landmarks are image-space, like every other observation here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request
import cv2
import numpy as np

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')
# Landmark indices from the MediaPipe hand topology.
FINGERTIPS = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
FRETTING_FINGERS = ('index', 'middle', 'ring', 'pinky')
CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))


def default_model_path():
    root = Path(os.environ.get('INLAY_CACHE', Path.home()/'.cache'/'inlay'))
    return root/'hand_landmarker.task'


def ensure_model(path=None):
    """Download the official hand landmarker bundle once, like the encoder weights."""
    path = Path(path) if path else default_model_path()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    if path.exists():
        raise ValueError(f'Hand landmarker bundle looks truncated: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading MediaPipe hand landmarker to {path}', flush=True)
    temporary = path.with_suffix('.tmp')
    urllib.request.urlretrieve(MODEL_URL, temporary)
    if temporary.stat().st_size < 1_000_000:
        temporary.unlink(missing_ok=True)
        raise ValueError('Downloaded hand landmarker bundle is too small')
    temporary.replace(path)
    return path


@dataclass
class Hand:
    """One detected hand. ``points`` are pixels; ``label`` is MediaPipe handedness."""
    points: np.ndarray
    label: str
    score: float

    def fingertip(self, finger):
        return self.points[FINGERTIPS[finger]]

    def fingertips(self, fingers=FRETTING_FINGERS):
        return {name: self.points[FINGERTIPS[name]] for name in fingers}


class HandTracker:
    """Video-mode hand landmarker. Timestamps must increase monotonically."""

    def __init__(self, model_path=None, num_hands=2, min_detection=.3, min_presence=.3,
                 min_tracking=.3):
        from mediapipe.tasks.python import vision, BaseOptions
        self._vision = vision
        options = vision.HandLandmarkerOptions(
            # The GPU delegate aborts in this desktop build; CPU is the supported path.
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path)),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def detect(self, frame, timestamp):
        """Return hands for a BGR frame. ``timestamp`` is seconds from media start."""
        import mediapipe as mp
        milliseconds = max(int(round(timestamp * 1000)), self.last_ms + 1)
        self.last_ms = milliseconds
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, milliseconds)
        height, width = frame.shape[:2]
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            points = np.array([[lm.x*width, lm.y*height] for lm in landmarks], np.float32)
            handedness = result.handedness[index][0] if index < len(result.handedness) else None
            hands.append(Hand(points, getattr(handedness, 'category_name', 'Unknown'),
                              float(getattr(handedness, 'score', 0.))))
        return hands

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fretting_hand(hands, observation):
    """Pick the hand on the neck: most landmarks inside the detected board.

    Handedness is not used. It is reported for the raw image, so a mirrored
    clip or a left-handed instrument would invert it silently.
    """
    if not hands or observation is None:
        return None
    neck = np.asarray(observation.neck, np.float32)
    best, best_score = None, 0
    for hand in hands:
        inside = sum(cv2.pointPolygonTest(neck, (float(x), float(y)), False) >= 0
                     for x, y in hand.points)
        if inside > best_score:
            best, best_score = hand, inside
    return best
