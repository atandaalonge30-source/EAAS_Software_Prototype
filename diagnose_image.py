import os
from eaas import ml_core
import cv2

CAP_DIR = os.path.join(os.path.dirname(__file__), 'eaas', 'static', 'captures')
files = sorted([f for f in os.listdir(CAP_DIR) if f.startswith('login_attempt')])
if not files:
    print('No login_attempt images found in', CAP_DIR)
    raise SystemExit(1)
fn = files[-1]
path = os.path.join(CAP_DIR, fn)
print('Testing image:', path)
img = cv2.imread(path)
if img is None:
    print('Could not read image')
    raise SystemExit(1)
roi, bbox, detected = ml_core.detect_face(img)
print('Detected:', detected)
print('BBox:', bbox)
print('ROI shape:', None if roi is None else roi.shape)
# annotate and save debug image
h, w = img.shape[:2]
x,y,ww,hh = bbox
color = (0,255,0) if detected else (0,0,255)
cv2.rectangle(img, (x,y), (x+ww, y+hh), color, 2)
debug_path = os.path.join(CAP_DIR, f'debug_{fn}')
cv2.imwrite(debug_path, img)
print('Wrote debug image to', debug_path)
