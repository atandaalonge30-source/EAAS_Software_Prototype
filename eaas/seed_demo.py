import requests
from synth_face import make_face, face_to_dataurl

BASE = "http://127.0.0.1:5055"

USERS = [
    dict(full_name="Oyewunmi Demilade Peter", matric_no="2021002681", department="Computer Science", email="demilade@lautech.edu.ng", skin=0),
    dict(full_name="Olalekan Yusuf Mayowa", matric_no="2021003180", department="Computer Science", email="yusuf@lautech.edu.ng", skin=1),
    dict(full_name="Bello Fatima Adaeze", matric_no="2021004452", department="Cyber Security", email="fatima@lautech.edu.ng", skin=2),
    dict(full_name="Chukwu Emeka John", matric_no="2021005810", department="Information Systems", email="emeka@lautech.edu.ng", skin=3),
]

user_ids = {}

for i, u in enumerate(USERS):
    frames = [
        face_to_dataurl(make_face(i + 1, emotion="neutral", jitter=j * 0.3, skin_idx=u["skin"]))
        for j in range(5)
    ]
    resp = requests.post(f"{BASE}/api/register", json={
        "full_name": u["full_name"], "matric_no": u["matric_no"],
        "department": u["department"], "email": u["email"], "frames": frames,
    })
    data = resp.json()
    print(u["full_name"], "->", data)
    if data.get("ok"):
        user_ids[i] = data["user_id"]

# Simulated login attempts: mix of genuine matches (neutral/happy),
# a stressed/angry attempt on an enrolled identity (-> flagged), and
# an unenrolled face (-> denied).
attempts = [
    (0, "neutral", 0.15),
    (0, "happy", 0.2),
    (1, "neutral", 0.15),
    (1, "sad", 0.2),
    (2, "happy", 0.15),
    (2, "angry", 0.2),
    (3, "neutral", 0.15),
    (3, "happy", 0.2),
    (0, "angry", 0.5),        # enrolled user, atypical + higher jitter -> extra verification
    (None, "neutral", 0.3),   # unenrolled identity -> denied
]

for idx, emotion, jitter in attempts:
    seed = (idx if idx is not None else 99) + 1
    skin = USERS[idx]["skin"] if idx is not None else 1
    frame = face_to_dataurl(make_face(seed, emotion=emotion, jitter=jitter, skin_idx=skin))
    resp = requests.post(f"{BASE}/api/login", json={"frame": frame})
    print("login attempt (user idx", idx, ", emotion", emotion, ") ->", resp.json())

print("\nSeeding complete.")
