# Emotion-Aware Authentication System (EAAS) — Prototype

## What this is
A working Flask web application implementing the system described in Chapter Four
of the project report: facial identity verification (OpenCV LBPH) fused with
real-time emotion recognition (scikit-learn ANN) through a weighted decision engine
that grants, denies, or flags a login attempt for additional verification.

## Requirements
Python 3.10+, then:
```
pip install flask opencv-contrib-python numpy scikit-learn joblib pillow --break-system-packages
```

## Setup & Run
```
python3 db.py                  # creates eaas.db
python3 train_emotion_model.py # trains the emotion classifier (models/)
python3 app.py                 # starts the server on http://127.0.0.1:5055
```
Open http://127.0.0.1:5055 in a browser with a webcam. Use "Register" to enrol a
user (captures 5 frames) and "Login / Scan" to authenticate.

## Demo data (optional)
`synth_face.py` and `seed_demo.py` generate illustrative stand-in faces and run
them through the real API to populate sample users/logs — useful only when no
webcam is available (e.g. for demonstration/screenshots). With a real webcam,
just use the Register/Login pages directly; you do not need these two files.

## Key files
- `app.py` — Flask routes
- `db.py` — SQLite schema/connection
- `ml_core.py` — face detection, LBPH identity recognizer, emotion feature
  extraction + ANN classifier, fusion decision engine
- `train_emotion_model.py` — builds the emotion training set and fits the model
- `templates/`, `static/` — UI
