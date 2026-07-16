"""
ml_core.py
Emotion-Aware Authentication System - Core Machine Learning Module
--------------------------------------------------------------------
This module implements the three core intelligent components described
in Chapter Four of the project:

    1. Facial Detection & Localization  -> detect_face()
    2. Facial Identification             -> FaceRecognizer (LBPH)
    3. Emotion Recognition                -> extract_emotion_features(),
                                             EmotionClassifier (MLP - scikit-learn)
    4. Fusion / Decision Making Unit      -> decide_access()

All processing is performed locally using OpenCV and scikit-learn so the
prototype can run on ordinary hardware (a standard PC/laptop webcam)
without dependence on external cloud services, consistent with the
scope defined in Chapter One (Section 1.5).
"""

import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = (160, 160)
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]

# Identity match is reported as a similarity percentage (100% = perfect
# match). LBPH internally returns a *distance* (lower = more similar);
# it is converted to a similarity score for readability in the UI/report.
IDENTITY_DISTANCE_CUTOFF = 90.0   # LBPH distance threshold (tuned empirically)
IDENTITY_MIN_SIMILARITY = 35.0    # % similarity required to accept identity (calibrated in Section 4.2.2 testing)

# Emotion 'risk' categories used by the decision engine. Emotions that
# commonly co-occur with duress, panic or unfamiliarity are flagged for
# extra scrutiny rather than an automatic hard denial, in line with the
# adaptive-security strategy discussed in Section 2.4.6.
RISK_EMOTIONS = {"Angry", "Surprised"}


# ---------------------------------------------------------------------
# 1. FACE DETECTION & LOCALIZATION
# ---------------------------------------------------------------------
def detect_face(bgr_image):
    """
    Detects the largest frontal face in the frame using a Haar-cascade
    classifier. If no face is confidently located (e.g. poor lighting,
    off-camera pose, or a non-facial capture), the routine falls back to
    a centre-weighted crop of the frame so that the pipeline remains
    fault tolerant, and flags the result accordingly.

    Returns: (face_roi_bgr, bbox, detected_flag)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    h, w = bgr_image.shape[:2]
    max_dim = int(min(h, w) * 0.62)  # a face rarely fills the whole capture frame

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5,
        minSize=(80, 80), maxSize=(max_dim, max_dim),
    )

    if len(faces) > 0:
        # choose the largest detected face (closest to camera)
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * fw)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
        roi = bgr_image[y0:y1, x0:x1]
        return roi, (x0, y0, x1 - x0, y1 - y0), True

    # Fallback: centre-weighted square crop (keeps pipeline operational
    # for degraded captures instead of hard-failing the request)
    side = int(min(h, w) * 0.8)
    cx, cy = w // 2, h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    roi = bgr_image[y0:y0 + side, x0:x0 + side]
    return roi, (x0, y0, side, side), False


def preprocess_face(roi_bgr):
    """Standardises a face ROI: grayscale, resize, histogram-equalise."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray


# ---------------------------------------------------------------------
# 2. FACIAL IDENTIFICATION (LBPH)
# ---------------------------------------------------------------------
class FaceRecognizer:
    """
    Wraps OpenCV's Local Binary Pattern Histogram (LBPH) face recognizer.
    LBPH was selected over heavier deep embedding models because it
    trains and predicts efficiently on CPU-only hardware, which matches
    the deployment environment defined in the project scope (standard
    PCs/laptops with webcams, Section 1.5).
    """

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=10, grid_y=10
        )
        self.label_map = {}   # int label -> user_id
        self.trained = False
        self.model_path = os.path.join(MODEL_DIR, "lbph_model.yml")
        self.map_path = os.path.join(MODEL_DIR, "label_map.npy")

    def train(self, samples):
        """samples: list of (user_id:int, gray_image np.ndarray)"""
        images, labels = [], []
        unique_ids = sorted(set(u for u, _ in samples))
        self.label_map = {i: uid for i, uid in enumerate(unique_ids)}
        inv_map = {uid: i for i, uid in self.label_map.items()}

        for uid, img in samples:
            images.append(img)
            labels.append(inv_map[uid])

        self.recognizer.train(images, np.array(labels))
        self.trained = True
        self.save()

    def predict(self, gray_face):
        if not self.trained:
            self.load()
        if not self.trained:
            return None, 0.0
        label, distance = self.recognizer.predict(gray_face)
        similarity = max(0.0, 100.0 - (distance / IDENTITY_DISTANCE_CUTOFF) * 100.0)
        similarity = min(similarity, 100.0)
        user_id = self.label_map.get(label)
        return user_id, round(similarity, 2)

    def save(self):
        self.recognizer.write(self.model_path)
        np.save(self.map_path, self.label_map)

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.map_path):
            self.recognizer.read(self.model_path)
            self.label_map = np.load(self.map_path, allow_pickle=True).item()
            self.trained = True


# ---------------------------------------------------------------------
# 3. EMOTION RECOGNITION
# ---------------------------------------------------------------------
def extract_emotion_features(gray_face):
    """
    Extracts a lightweight, hand-engineered geometric/textural feature
    vector describing muscle-movement cues in three facial sub-regions
    (brow, eye, mouth), following the region-based analysis approach
    common in real-time affective computing systems (Section 2.4.4).

    For each region the feature set captures:
        - mean intensity (shadow/highlight cues from muscle contraction)
        - Sobel gradient energy (edge sharpness -> wrinkling/furrowing)
        - vertical gradient bias (upward vs downward curvature, used to
          separate expressions such as smiling from frowning)

    This 9-dimensional vector is passed to the MLP classifier below.
    """
    h, w = gray_face.shape
    brow = gray_face[int(h * 0.14):int(h * 0.30), :]
    eyes = gray_face[int(h * 0.30):int(h * 0.55), :]
    mouth = gray_face[int(h * 0.82):h, :]

    feats = []
    for region in (brow, eyes, mouth):
        region_f = region.astype(np.float32)
        mean_i = region_f.mean() / 255.0
        sobel_y = cv2.Sobel(region_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = float(np.mean(np.abs(sobel_y))) / 255.0
        vertical_bias = float(np.mean(sobel_y)) / 255.0
        feats.extend([mean_i, grad_energy, vertical_bias])

    return np.array(feats, dtype=np.float32)


class EmotionClassifier:
    """
    A lightweight Multi-Layer Perceptron (Artificial Neural Network)
    trained with scikit-learn on the geometric feature representation
    above. A compact ANN was chosen (rather than a full convolutional
    network trained on raw pixels) to keep inference time low enough
    for real-time authentication on standard hardware, mirroring the
    lightweight-network design strategy discussed in the literature
    (Li et al., 2022) and reviewed in Chapter Two.
    """

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=2000,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.model_path = os.path.join(MODEL_DIR, "emotion_mlp.joblib")
        self.scaler_path = os.path.join(MODEL_DIR, "emotion_scaler.joblib")
        self.trained = False

    def train(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.trained = True

    def predict(self, feature_vector):
        if not self.trained:
            self.load()
        if not self.trained:
            return "Neutral", 0.0
        Xs = self.scaler.transform([feature_vector])
        probs = self.model.predict_proba(Xs)[0]
        idx = int(np.argmax(probs))
        label = self.model.classes_[idx]
        return label, round(float(probs[idx]) * 100, 2)


def bootstrap_emotion_training_set(n_per_class=60, seed=42):
    """
    Generates a labelled synthetic feature-training set that reflects
    the characteristic feature signatures of each emotion class in the
    region-based representation used by extract_emotion_features().
    This mirrors the documented data-collection phase (Section 1.6.2 -
    primary data captured under varied facial expressions) and gives
    the classifier a consistent, reproducible starting model before it
    is further fine-tuned on organisation-specific captured samples.
    """
    rng = np.random.RandomState(seed)
    X, y = [], []
    profiles = {
        "Neutral":   dict(brow=(0.55, 0.10, 0.00), eyes=(0.50, 0.12, 0.00), mouth=(0.45, 0.10, 0.00)),
        "Happy":     dict(brow=(0.55, 0.10, 0.00), eyes=(0.48, 0.15, -0.02), mouth=(0.55, 0.30, -0.18)),
        "Sad":       dict(brow=(0.50, 0.14, 0.05), eyes=(0.45, 0.12, 0.02), mouth=(0.40, 0.20, 0.15)),
        "Angry":     dict(brow=(0.42, 0.28, 0.10), eyes=(0.40, 0.22, 0.05), mouth=(0.38, 0.18, 0.05)),
        "Surprised": dict(brow=(0.62, 0.30, -0.15), eyes=(0.55, 0.25, -0.10), mouth=(0.50, 0.35, -0.05)),
    }
    for label, regions in profiles.items():
        for _ in range(n_per_class):
            row = []
            for region_key in ("brow", "eyes", "mouth"):
                mean_i, energy, bias = regions[region_key]
                row.append(mean_i + rng.normal(0, 0.04))
                row.append(max(0.0, energy + rng.normal(0, 0.03)))
                row.append(bias + rng.normal(0, 0.03))
            X.append(row)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------
# 4. FUSION / DECISION-MAKING UNIT
# ---------------------------------------------------------------------
def decide_access(identity_conf, emotion_label, emotion_conf, baseline_emotion=None):
    """
    Weighted fusion of the identity and emotion verification scores,
    implementing the strategy described in Section 2.4.6. Identity
    verification carries the greater weight; emotional analysis
    supplies supplementary, adaptive validation rather than a hard
    veto, so that a genuine but momentarily stressed user is not
    unfairly locked out.
    """
    if identity_conf < IDENTITY_MIN_SIMILARITY:
        return "DENIED", "Identity not recognised", "danger"

    deviates_from_baseline = (
        baseline_emotion is not None and emotion_label != baseline_emotion
    )

    if emotion_label in RISK_EMOTIONS and (emotion_conf >= 60 or deviates_from_baseline):
        return (
            "ADDITIONAL VERIFICATION REQUIRED",
            f"Identity confirmed, but an atypical emotional state "
            f"({emotion_label}) was detected during login",
            "warning",
        )

    return "GRANTED", "Identity and emotional pattern consistent with profile", "success"
