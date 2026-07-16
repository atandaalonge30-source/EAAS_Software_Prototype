"""
synth_face.py
Generates simple, clearly-illustrative stand-in face images used ONLY to
exercise and demonstrate the working pipeline end-to-end in this sandbox
(no camera hardware / no internet access to source volunteer photographs
is available in this build/demo environment). In a real deployment, these
frames are replaced one-for-one by live webcam captures via getUserMedia()
as implemented in static/js/capture.js and templates/register.html /
login.html.
"""
import random
import base64
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SKIN_TONES = [(222, 184, 150), (198, 152, 116), (141, 100, 71), (240, 205, 170)]


def make_face(identity_seed, emotion="neutral", jitter=0.0, skin_idx=0):
    EMO_SALT = {'neutral':11,'happy':23,'sad':37,'angry':53,'surprised':71}
    rng = random.Random(identity_seed * 97 + EMO_SALT.get(emotion,0))
    id_rng = random.Random(identity_seed * 131)  # identity-only randomness (stable across emotions)
    W, H = 480, 480
    img = Image.new("RGB", (W, H), (20, 24, 33))
    d = ImageDraw.Draw(img)
    skin = SKIN_TONES[skin_idx % len(SKIN_TONES)]
    cx, cy = W // 2, H // 2 + 10
    jx = int(rng.uniform(-8, 8) * (1 + jitter))
    jy = int(rng.uniform(-8, 8) * (1 + jitter))
    cx += jx
    cy += jy

    # identity-specific face shape (stable per identity, independent of emotion/jitter)
    width_scale = 0.85 + 0.3 * id_rng.random()
    height_scale = 0.85 + 0.3 * id_rng.random()
    rw, rh = int(125 * width_scale), int(160 * height_scale)
    hairstyle = id_rng.choice(["cap", "side", "full"])
    has_glasses = id_rng.random() < 0.5
    has_mark = True
    mark_pos = (id_rng.uniform(-0.6, 0.6), id_rng.uniform(-0.2, 0.6))

    d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=skin)

    # identity-specific speckled skin micro-texture (gives LBPH real
    # per-identity textural signal, akin to distinguishing skin/hair
    # micro-patterns captured in real photographs)
    tex_rng = random.Random(identity_seed * 733)
    for _ in range(380):
        tx = cx + tex_rng.uniform(-rw * 0.85, rw * 0.85)
        ty = cy + tex_rng.uniform(-rh * 0.85, rh * 0.85)
        shade = tex_rng.choice([-30, -22, -14, 12, 20, 28])
        c = tuple(max(0, min(255, ch + shade)) for ch in skin)
        d.ellipse([tx - 2, ty - 2, tx + 2, ty + 2], fill=c)

    # hairstyle (identity-specific, stable across emotion/jitter)
    hair_color = (25 + id_rng.randint(0, 30), 18 + id_rng.randint(0, 20), 12 + id_rng.randint(0, 15))
    if hairstyle == "cap":
        d.pieslice([cx - rw - 5, cy - rh - 40, cx + rw + 5, cy - 20], 180, 360, fill=hair_color)
    elif hairstyle == "side":
        d.pieslice([cx - rw - 5, cy - rh - 55, cx + rw + 5, cy - 10], 175, 350, fill=hair_color)
        d.rectangle([cx - rw - 5, cy - rh + 10, cx - rw + 18, cy + 40], fill=hair_color)
    elif hairstyle == "full":
        d.pieslice([cx - rw - 12, cy - rh - 60, cx + rw + 12, cy + 10], 170, 370, fill=hair_color)
    # bald: no hair drawn

    if has_mark:
        mx = cx + mark_pos[0] * rw
        my = cy + mark_pos[1] * rh
        d.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=(70, 40, 35))

    eye_y = cy - 30
    for side in (-1, 1):
        ex = cx + side * 55
        d.ellipse([ex - 22, eye_y - 14, ex + 22, eye_y + 14], fill=(250, 250, 248))
        pupil_shift = -6 if emotion == "surprised" else 0
        d.ellipse([ex - 10, eye_y - 10 + pupil_shift, ex + 10, eye_y + 10 + pupil_shift], fill=(50, 35, 25))

    # eyebrows
    for side in (-1, 1):
        ex = cx + side * 55
        if emotion == "angry":
            y0, y1 = (eye_y - 32, eye_y - 16) if side < 0 else (eye_y - 16, eye_y - 32)
        elif emotion == "surprised":
            y0 = y1 = eye_y - 38
        elif emotion == "sad":
            y0, y1 = (eye_y - 20, eye_y - 30) if side < 0 else (eye_y - 30, eye_y - 20)
        else:
            y0, y1 = eye_y - 26, eye_y - 24
        d.line([ex - 24, y0, ex + 24, y1], fill=(35, 24, 17), width=6)

    if has_glasses:
        for side in (-1, 1):
            ex = cx + side * 55
            d.ellipse([ex - 24, eye_y - 17, ex + 24, eye_y + 17], outline=(15, 15, 15), width=3)
        d.line([cx - 8, eye_y, cx + 8, eye_y], fill=(15, 15, 15), width=3)

    # nose
    d.line([cx, cy - 5, cx - 7, cy + 25], fill=tuple(max(0, c - 35) for c in skin), width=3)
    d.line([cx - 7, cy + 25, cx + 9, cy + 25], fill=tuple(max(0, c - 35) for c in skin), width=3)

    # mouth
    my = cy + 65
    mouth_color = (140, 55, 55)
    if emotion == "happy":
        d.arc([cx - 42, my - 26, cx + 42, my + 14], 15, 165, fill=mouth_color, width=7)
    elif emotion == "sad":
        d.arc([cx - 40, my - 2, cx + 40, my + 38], 200, 340, fill=mouth_color, width=7)
    elif emotion == "surprised":
        d.ellipse([cx - 18, my - 12, cx + 18, my + 24], fill=(90, 30, 30))
    elif emotion == "angry":
        d.line([cx - 34, my + 8, cx + 34, my - 4], fill=mouth_color, width=7)
        d.line([cx - 20, my - 30, cx + 20, my - 34], fill=(35, 24, 17), width=3)
    else:
        d.line([cx - 34, my, cx + 34, my], fill=mouth_color, width=6)

    img = img.filter(ImageFilter.GaussianBlur(1.2))
    return img


def face_to_dataurl(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
