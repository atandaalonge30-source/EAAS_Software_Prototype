import sqlite3, os, sys
try:
    from eaas import ml_core
except Exception as e:
    print('ERROR importing ml_core:', e)
    sys.exit(1)
print('HAAR_CASCADE_PATH=', ml_core.HAAR_CASCADE_PATH)
print('FACE_CASCADE.empty()=', ml_core.FACE_CASCADE.empty())
DB='eaas/eaas.db'
if os.path.exists(DB):
    conn=sqlite3.connect(DB)
    conn.row_factory=sqlite3.Row
    rows=conn.execute('SELECT id,face_detected,decision,reason,identity_confidence,emotion_label,emotion_confidence,image_path,created_at FROM access_logs ORDER BY id DESC LIMIT 10').fetchall()
    print('Recent access_logs:')
    for r in rows:
        print(dict(r))
else:
    print('DB not found at', DB)
