#!/usr/bin/env python3
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

url = 'https://eaas-software-prototype-8oj6.onrender.com/api/login'
img_path = Path('eaas/demo_cam_face.png')
if not img_path.exists():
    raise SystemExit(f'Missing sample image: {img_path}')

with img_path.open('rb') as f:
    data = f.read()
frame_data = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
req = urllib.request.Request(url, data=json.dumps({'frame': frame_data}).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    r = urllib.request.urlopen(req, timeout=20)
    print('POST /api/login ->', r.getcode())
    print(r.read().decode('utf-8', 'ignore'))
except urllib.error.HTTPError as e:
    body = e.read().decode('utf-8', 'ignore')
    print('HTTPError', e.code)
    print(body)
except Exception as e:
    print('Request failed:', repr(e))
