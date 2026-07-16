"""
train_emotion_model.py
Builds the emotion-classification training set by running labelled sample
faces through the ACTUAL detection -> preprocessing -> feature-extraction
pipeline used at inference time (ml_core.detect_face / extract_emotion_features),
then fits the scikit-learn MLP classifier on the resulting feature vectors.

In a full production deployment this script would be pointed at a folder
of labelled volunteer photographs collected during the data-collection
phase (Section 1.6.2). For this build/demo environment (no camera/internet
available in the sandbox), synth_face.py supplies the labelled samples so
that the complete training pipeline can be exercised and verified end to
end.
"""
import numpy as np
import cv2

import ml_core
from synth_face import make_face

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]


def build_dataset(samples_per_class=50):
    X, y = [], []
    for emo in EMOTIONS:
        for i in range(samples_per_class):
            seed = i + 1
            jitter = (i % 5) * 0.15
            skin = i % 4
            img = make_face(seed, emotion=emo.lower(), jitter=jitter, skin_idx=skin)
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            roi, bbox, detected = ml_core.detect_face(bgr)
            gray = ml_core.preprocess_face(roi)
            feats = ml_core.extract_emotion_features(gray)
            X.append(feats)
            y.append(emo)
    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    X, y = build_dataset()
    print("Training set shape:", X.shape)

    clf = ml_core.EmotionClassifier()
    clf.train(X, y)

    # quick holdout check on freshly generated, unseen samples
    Xte, yte = build_dataset(samples_per_class=20)
    preds = [clf.predict(f)[0] for f in Xte]
    acc = np.mean(np.array(preds) == yte)
    print(f"Holdout accuracy on unseen synthetic samples: {acc*100:.2f}%")

    from collections import Counter
    print("Prediction distribution:", Counter(preds))
