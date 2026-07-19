import os
import base64
import sqlite3
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, redirect

from .db import get_conn, init_db, DB_PATH
from .ml_core import (
    detect_face, preprocess_face, FaceRecognizer, EmotionClassifier,
    extract_emotion_features, bootstrap_emotion_training_set, decide_access,
    IDENTITY_MIN_SIMILARITY,
)


def ml_core_min_similarity():
    return IDENTITY_MIN_SIMILARITY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

app = Flask(__name__)

face_recognizer = FaceRecognizer()
emotion_clf = EmotionClassifier()


def ensure_bootstrapped():
    if not os.path.exists(DB_PATH):
        init_db()
    if not emotion_clf.trained:
        emotion_clf.load()
    if not emotion_clf.trained:
        X, y = bootstrap_emotion_training_set()
        emotion_clf.train(X, y)
    face_recognizer.load()


def decode_frame(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_capture(bgr_image, prefix):
    fname = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    path = os.path.join(CAPTURE_DIR, fname)
    cv2.imwrite(path, bgr_image)
    return fname


def find_existing_face_match(conn, gray_faces, similarity_threshold=78.0):
    if not gray_faces:
        return None
    if not face_recognizer.trained:
        retrain_face_recognizer(conn)
    if not face_recognizer.trained:
        return None

    for gray_face in gray_faces:
        user_id, similarity = face_recognizer.predict(gray_face)
        if user_id is not None and similarity >= similarity_threshold:
            return user_id
    return None


from contextlib import suppress

def delete_access_log(conn, log_id):
    if not (row := conn.execute("SELECT image_path FROM access_logs WHERE id=?", (log_id,)).fetchone()):
        return False

    conn.execute("DELETE FROM access_logs WHERE id=?", (log_id,))
    image_path = row["image_path"]
    if image_path:
        with suppress(FileNotFoundError, OSError):
            os.remove(os.path.join(CAPTURE_DIR, image_path))
    return True


def delete_user(conn, user_id):
    if not (user := conn.execute("SELECT photo_path FROM users WHERE id=?", (user_id,)).fetchone()):
        return False

    conn.execute("DELETE FROM access_logs WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))

    if user["photo_path"]:
        with suppress(FileNotFoundError, OSError):
            os.remove(os.path.join(CAPTURE_DIR, user["photo_path"]))

    for fname in os.listdir(CAPTURE_DIR):
        if fname.startswith(f"user{user_id}_enrol"):
            with suppress(FileNotFoundError, OSError):
                os.remove(os.path.join(CAPTURE_DIR, fname))
    return True


# ---------------------------------------------------------------------
@app.route("/")
def home():
    conn = get_conn()
    user_count = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    log_count = conn.execute("SELECT COUNT(*) c FROM access_logs").fetchone()["c"]
    granted = conn.execute(
        "SELECT COUNT(*) c FROM access_logs WHERE decision='GRANTED'"
    ).fetchone()["c"]
    granted_pct = round((granted / log_count) * 100, 1) if log_count else 0
    conn.close()
    return render_template(
        "index.html", active="home",
        user_count=user_count, log_count=log_count, granted_pct=granted_pct,
    )


@app.route("/register")
def register():
    return render_template("register.html", active="register")


@app.route("/re-enrol/<int:user_id>")
def re_enrol(user_id):
    conn = get_conn()
    u = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not u:
        return redirect(url_for("admin_users"))
    return render_template("re_enrol.html", active="users", user=u)


@app.route("/login")
def login():
    return render_template("login.html", active="login")


@app.route("/api/register", methods=["POST"])
def api_register():
    payload = request.get_json()
    full_name = payload.get("full_name", "").strip()
    matric_no = payload.get("matric_no", "").strip()
    department = payload.get("department", "").strip()
    email = payload.get("email", "").strip()
    frames_b64 = payload.get("frames", [])

    if not full_name or not matric_no or not frames_b64:
        return jsonify(ok=False, error="Missing required enrolment data"), 400

    conn = get_conn()

    # process captured frames once so we can reject duplicate faces before creating a new account
    gray_faces = []
    emotion_feats = []
    face_rois = []
    for fb64 in frames_b64:
        bgr = decode_frame(fb64)
        roi, bbox, detected = detect_face(bgr)
        gray = preprocess_face(roi)
        gray_faces.append(gray)
        emotion_feats.append(extract_emotion_features(gray))
        face_rois.append(roi)

    existing_user_id = find_existing_face_match(conn, gray_faces)
    if existing_user_id is not None:
        conn.close()
        return jsonify(ok=False, error="This face is already enrolled in the system."), 409

    try:
        cur = conn.execute(
            "INSERT INTO users (full_name, matric_no, department, email) VALUES (?,?,?,?)",
            (full_name, matric_no, department, email),
        )
        user_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify(ok=False, error="Matric number already registered."), 400
    except Exception as e:
        conn.close()
        return jsonify(ok=False, error=f"Could not create user: {e}"), 400

    representative_photo = None
    for i, roi in enumerate(face_rois):
        fname = save_capture(roi, f"user{user_id}_enrol{i}")
        if representative_photo is None:
            representative_photo = fname

    conn.execute(
        "UPDATE users SET photo_path=? WHERE id=?", (representative_photo, user_id)
    )

    # baseline emotion = majority vote across enrolment captures
    labels = [emotion_clf.predict(f)[0] for f in emotion_feats]
    baseline = max(set(labels), key=labels.count) if labels else "Neutral"
    conn.execute("UPDATE users SET baseline_emotion=? WHERE id=?", (baseline, user_id))
    conn.commit()

    # retrain LBPH recognizer across ALL enrolled users' stored samples
    retrain_face_recognizer(conn)
    conn.close()

    return jsonify(ok=True, user_id=user_id)


@app.route("/api/re_enrol/<int:user_id>", methods=["POST"])
def api_re_enrol(user_id):
    payload = request.get_json()
    frames_b64 = payload.get("frames", [])
    if not frames_b64:
        return jsonify(ok=False, error="No frames received"), 400

    conn = get_conn()
    if not (u := conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()):
        conn.close()
        return jsonify(ok=False, error="User not found"), 404

    # remove old enrolment files for this user so we start fresh
    for fname in os.listdir(CAPTURE_DIR):
        if fname.startswith(f"user{user_id}_enrol"):
            try:
                os.remove(os.path.join(CAPTURE_DIR, fname))
            except Exception:
                pass

    representative_photo = None
    emotion_feats = []
    for i, fb64 in enumerate(frames_b64):
        bgr = decode_frame(fb64)
        roi, bbox, detected = detect_face(bgr)
        gray = preprocess_face(roi)
        emotion_feats.append(extract_emotion_features(gray))
        fname = save_capture(roi, f"user{user_id}_enrol{i}")
        if representative_photo is None:
            representative_photo = fname

    conn.execute("UPDATE users SET photo_path=? WHERE id=?", (representative_photo, user_id))

    labels = [emotion_clf.predict(f)[0] for f in emotion_feats]
    baseline = max(set(labels), key=labels.count) if labels else u["baseline_emotion"]
    conn.execute("UPDATE users SET baseline_emotion=? WHERE id=?", (baseline, user_id))
    conn.commit()

    # retrain LBPH recognizer across ALL enrolled users' stored samples
    retrain_face_recognizer(conn)
    conn.close()

    return jsonify(ok=True, user_id=user_id)


def retrain_face_recognizer(conn):
    """Rebuild the LBPH model from every stored capture on disk (users table
    photo_path is the representative photo; we also glob per-user enrolment
    captures saved during /api/register for a richer training set)."""
    users = conn.execute("SELECT id FROM users").fetchall()
    samples = []
    for u in users:
        uid = u["id"]
        prefix = f"user{uid}_enrol"
        for fname in os.listdir(CAPTURE_DIR):
            if fname.startswith(prefix):
                img = cv2.imread(os.path.join(CAPTURE_DIR, fname))
                if img is not None:
                    samples.append((uid, preprocess_face(img)))
    if samples:
        face_recognizer.train(samples)


@app.route("/register/success/<int:user_id>")
def register_success(user_id):
    conn = get_conn()
    u = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    image_url = url_for("static", filename=f"captures/{u['photo_path']}")
    return render_template(
        "register_success.html", active="register",
        full_name=u["full_name"], matric_no=u["matric_no"], image_url=image_url,
    )


@app.route("/api/login", methods=["POST"])
def api_login():
    payload = request.get_json()
    frame_b64 = payload.get("frame")
    if not frame_b64:
        return jsonify(ok=False, error="No frame received"), 400

    try:
        bgr = decode_frame(frame_b64)
        roi, bbox, detected = detect_face(bgr)
        gray = preprocess_face(roi)

        user_id, identity_conf = face_recognizer.predict(gray)
        feats = extract_emotion_features(gray)
        emotion_label, emotion_conf = emotion_clf.predict(feats)
    except Exception as exc:
        # Log and return a readable error if inference fails.
        return jsonify(ok=False, error=f"Inference failed: {exc}"), 500

    conn = get_conn()
    name = "Unknown"
    baseline = None
    matched_user_id = None
    if user_id is not None and identity_conf >= ml_core_min_similarity():
        u = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if u:
            name = u["full_name"]
            baseline = u["baseline_emotion"]
            matched_user_id = user_id

    decision, reason, level = decide_access(identity_conf, emotion_label, emotion_conf, baseline, detected)
    fname = save_capture(roi, "login_attempt")

    cur = conn.execute(
        """INSERT INTO access_logs
           (user_id, attempt_name, identity_confidence, emotion_label,
            emotion_confidence, decision, reason, image_path, face_detected)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (matched_user_id, name, identity_conf, emotion_label, emotion_conf,
         decision, reason, fname, int(detected)),
    )
    log_id = cur.lastrowid
    conn.commit()
    conn.close()

    return jsonify(ok=True, log_id=log_id)


@app.route("/result/<int:log_id>")
def result(log_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM access_logs WHERE id=?", (log_id,)).fetchone()
    conn.close()
    if row is None:
        return "Result not found", 404

    level = {"SUCCESS": "success", "GRANTED": "success", "ADDITIONAL VERIFICATION REQUIRED": "warning"}.get(
        row["decision"], "danger"
    )
    image_url = url_for("static", filename=f"captures/{row['image_path']}")
    return render_template(
        "result.html", active="login",
        decision=row["decision"], reason=row["reason"], level=level,
        identity_name=row["attempt_name"], identity_conf=row["identity_confidence"],
        emotion_label=row["emotion_label"], emotion_conf=row["emotion_confidence"],
        image_url=image_url,
    )


@app.route("/admin/logs")
def admin_logs():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM access_logs ORDER BY id DESC"
    ).fetchall()
    conn.close()
    level_map = {"SUCCESS": "success", "GRANTED": "success", "ADDITIONAL VERIFICATION REQUIRED": "warning"}
    logs = []
    total = granted = flagged = denied = 0
    for r in rows:
        total += 1
        lvl = level_map.get(r["decision"], "danger")
        if r["decision"] in ("GRANTED", "SUCCESS"):
            granted += 1
        elif lvl == "warning":
            flagged += 1
        else:
            denied += 1
        logs.append({
            "id": r["id"],
            "image_url": url_for("static", filename=f"captures/{r['image_path']}"),
            "name": r["attempt_name"],
            "identity_confidence": r["identity_confidence"],
            "emotion_label": r["emotion_label"],
            "emotion_confidence": r["emotion_confidence"],
            "decision": r["decision"],
            "level": lvl,
            "created_at": r["created_at"],
        })
    return render_template(
        "admin_logs.html", active="logs", logs=logs,
        total=total, granted=granted, flagged=flagged, denied=denied,
    )


@app.route("/admin/logs/delete/<int:log_id>", methods=["POST"])
def delete_log(log_id):
    conn = get_conn()
    deleted = delete_access_log(conn, log_id)
    conn.commit()
    conn.close()
    return redirect(url_for("admin_logs")) if deleted else redirect(url_for("admin_logs"))


@app.route("/admin/model")
def admin_model():
    # show status of face & emotion models and label map
    status = {
        "face_trained": bool(face_recognizer.trained),
        "label_map_size": len(face_recognizer.label_map) if hasattr(face_recognizer, 'label_map') else 0,
        "emotion_trained": bool(emotion_clf.trained),
    }
    # prepare label map list for display
    label_map = [
        {"label": int(label), "user_id": int(uid)}
        for label, uid in face_recognizer.label_map.items()
    ] if hasattr(face_recognizer, 'label_map') and isinstance(face_recognizer.label_map, dict) else []
    return render_template("admin_model.html", active="logs", status=status, label_map=label_map)


@app.route("/admin/model/retrain", methods=["POST"])
def retrain_model_route():
    conn = get_conn()
    # retrain face recognizer from disk
    retrain_face_recognizer(conn)
    # re-bootstrap emotion classifier if needed
    if not emotion_clf.trained:
        X, y = bootstrap_emotion_training_set()
        emotion_clf.train(X, y)
    conn.close()
    return redirect(url_for("admin_model"))


@app.route("/admin/users")
def admin_users():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    conn.close()
    users = [{
        "id": r["id"],
        "image_url": url_for("static", filename=f"captures/{r['photo_path']}"),
        "full_name": r["full_name"], "matric_no": r["matric_no"],
        "department": r["department"], "baseline_emotion": r["baseline_emotion"],
        "created_at": r["created_at"],
    } for r in rows]
    return render_template("admin_users.html", active="users", users=users)


@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
def delete_user_route(user_id):
    conn = get_conn()
    delete_user(conn, user_id)
    conn.commit()
    conn.close()
    return redirect(url_for("admin_users"))


ensure_bootstrapped()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=False)
