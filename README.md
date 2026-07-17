# Emotion-Aware Authentication System (EAAS) - Prototype

A working Flask web application implementing facial identity verification (OpenCV LBPH) fused with real-time emotion recognition (scikit-learn MLP) through an intelligent decision engine for secure authentication.

## 🎯 Features

- **Face Detection & Capture**: Real-time facial detection using Haar Cascade classifiers
- **Emotion Recognition**: Detects emotions (Happy, Sad, Angry, Surprised, Neutral) using handcrafted features and neural networks
- **Facial Identity Verification**: LBPH (Local Binary Pattern Histogram) face recognition for identity matching
- **User Enrollment**: Register new users with multi-frame facial data
- **Secure Login**: Face-based authentication with emotion analysis
- **Admin Dashboard**: View access logs, manage users, and monitor system status
- **Local Processing**: All ML models run locally without external cloud dependencies

## 📋 Requirements

- Python 3.10+
- Webcam for face capture
- 50MB disk space for models and database

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd EAAS_Software_Prototype
python -m venv .venv

# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Initialize Database & Train Models

```bash
python -m eaas.db              # Create SQLite database
python -m eaas.train_emotion_model  # Train emotion classifier
```

### 3. Run the Application

```bash
python -m eaas.app
```

Open http://127.0.0.1:5055 in your browser.

## 📱 How to Use

### Register a New User

1. Click "Register" on the home page
2. Enter full name, matric number, department, and email
3. Allow camera access
4. Capture 5 facial frames by positioning your face in the center
5. Click "Enroll" to save your profile
6. You'll receive a confirmation with your enrolled photo

### Login / Authenticate

1. Click "Login / Scan" on the home page
2. Allow camera access
3. Position your face in the center of the capture area
4. The system will automatically detect your face
5. You'll receive a report showing:
   - **Face Detection**: Whether your face was successfully captured
   - **Emotion**: Detected emotion (Happy, Sad, Angry, Surprised, or Neutral)
   - **Confidence**: Emotion detection confidence score

### View Admin Dashboard

1. Click "Admin Logs" to view all authentication attempts
2. Click "Admin Users" to manage enrolled users
3. Click "Model Status" to check system configuration

## 📊 System Architecture

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Face Detection** | OpenCV Haar Cascade | Real-time facial detection |
| **Identity Recognition** | OpenCV LBPH | Facial biometric matching |
| **Emotion Analysis** | scikit-learn MLP | Affective state classification |
| **Backend** | Flask | REST API & web framework |
| **Database** | SQLite | User profiles & access logs |
| **Frontend** | HTML/CSS/JS | Web interface with video capture |

### Decision Engine

The system combines identity confidence and emotion analysis to make authentication decisions:

- **✅ SUCCESS**: Face detected + emotion classification
- **❌ DENIED**: No face detected
- **⚠️ ADDITIONAL VERIFICATION**: Identity confirmed but emotional state differs from baseline

## 📦 Project Structure

```
EAAS_Software_Prototype/
├── eaas/
│   ├── app.py                      # Flask application & routes
│   ├── db.py                       # Database initialization
│   ├── ml_core.py                  # ML pipeline (face detection, emotion, identity)
│   ├── train_emotion_model.py     # Emotion classifier training
│   ├── data/
│   │   └── haarcascade_frontalface_default.xml
│   ├── models/                     # Pre-trained ML models
│   │   ├── lbph_model.yml
│   │   ├── emotion_mlp.joblib
│   │   └── emotion_scaler.joblib
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/capture.js
│   │   └── captures/               # Captured face images
│   ├── templates/                  # HTML pages
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── register.html
│   │   ├── login.html
│   │   ├── result.html
│   │   ├── admin_logs.html
│   │   ├── admin_users.html
│   │   └── admin_model.html
│   └── eaas.db                     # SQLite database (auto-generated)
├── Procfile                        # Heroku/Render deployment config
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore rules
```

## 🌐 Deployment

### Deploy on Render

1. **Fork or Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: EAAS Prototype"
   git push origin main
   ```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch
   - **Build Command**: `pip install -r requirements.txt && python -m eaas.db`
   - **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT eaas.app:app`

4. **Configure Environment**
   - Add environment variable if needed
   - Click "Deploy"

5. **Access Your Application**
   - Your app will be available at `https://<service-name>.onrender.com`

### Deploy on Heroku

```bash
heroku create <app-name>
git push heroku main
heroku config:set FLASK_ENV=production
```

## 🔧 Configuration

### Tuning Face Detection

Edit `ml_core.py` to adjust detection sensitivity:

```python
IDENTITY_MIN_SIMILARITY = 35.0    # Lower = more permissive
IDENTITY_DISTANCE_CUTOFF = 90.0   # LBPH distance threshold
```

### Emotion Confidence Threshold

Modify the emotion classification logic in `decide_access()` function.

## 📈 Monitoring

### Access Logs

All authentication attempts are logged with:
- Identity confidence score
- Detected emotion
- Emotion confidence
- Decision outcome
- Timestamp
- Captured image

### Database Management

To view logs directly:

```python
import sqlite3
conn = sqlite3.connect('eaas/eaas.db')
c = conn.cursor()
c.execute("SELECT * FROM access_logs ORDER BY created_at DESC LIMIT 10")
for row in c.fetchall():
    print(row)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check browser permissions, try different browser |
| "No face detected" | Ensure good lighting, position face in center |
| Poor emotion accuracy | Register more users, retrain emotion model |
| Database errors | Delete `eaas.db` and re-run `python -m eaas.db` |
| Models not loading | Ensure `train_emotion_model.py` has run successfully |

## 📝 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Home page with statistics |
| `/register` | GET | Registration form |
| `/api/register` | POST | Submit registration |
| `/login` | GET | Login page |
| `/api/login` | POST | Submit face capture for authentication |
| `/result/<id>` | GET | View authentication result |
| `/admin/logs` | GET | View all access logs |
| `/admin/users` | GET | Manage enrolled users |
| `/admin/model` | GET | View model status |

## 🔐 Security Notes

- All face images are stored locally
- No data is sent to external servers
- Database contains hashed/encoded biometric templates (LBPH features)
- Consider using HTTPS in production
- Implement rate limiting for login attempts

## 📄 License

This project is part of the academic research on Emotion-Aware Authentication Systems.

## 👤 Author

Developed as a prototype for emotion-aware biometric authentication research.

## 💬 Support

For issues or questions, please open a GitHub issue or contact the development team.
