# debug_plate.py — Run this standalone to diagnose what the camera sees
# Usage: python debug_plate.py
# Press keys in the window:
#   s = save current frame to debug_frame.jpg
#   f = force OCR on full frame (bypass contour detection)
#   q = quit

import os, logging, sys

def get_app_base_dir() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_base_dir()
RUN_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else APP_DIR

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(RUN_DIR, ".paddlex"))
os.makedirs(os.environ["PADDLE_PDX_CACHE_HOME"], exist_ok=True)
logging.disable(logging.WARNING)

import cv2
import numpy as np
import re
import time
from paddleocr import PaddleOCR

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(APP_DIR, "config.ini"))

entry_camera_url = config.get("Cameras", "entry_camera_url")
try:
    entry_camera_url = int(entry_camera_url)
except ValueError:
    pass

def create_ocr_engine():
    try:
        return PaddleOCR(use_textline_orientation=True, lang='en')
    except (TypeError, ValueError):
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

print("[DEBUG] Loading OCR engine...")
ocr = create_ocr_engine()
print("[DEBUG] OCR engine ready.")

if isinstance(entry_camera_url, int) and sys.platform.startswith("win"):
    cap = cv2.VideoCapture(entry_camera_url, cv2.CAP_DSHOW)
elif isinstance(entry_camera_url, int) and sys.platform == "darwin":
    cap = cv2.VideoCapture(entry_camera_url, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(entry_camera_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(config.get("General", "camera_width")))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.get("General", "camera_height")))

print("[DEBUG] Camera opened. Press:")
print("  s = save frame to disk")
print("  f = force OCR on FULL frame right now")
print("  q = quit")


def run_ocr_full(img, label="", conf_thresh=0.2):
    """OCR the full image and print every result regardless of confidence."""
    result = ocr.ocr(img, cls=True)
    print(f"\n--- OCR [{label}] ---")
    if not result or result[0] is None:
        print("  (no result)")
        return [], []
    boxes_to_draw = []
    texts = []
    for line in result[0]:
        text = line[1][0].upper()
        conf = line[1][1]
        box  = line[0]
        print(f"  conf={conf:.2f}  text='{text}'")
        boxes_to_draw.append((box, text, conf))
        texts.append(text)
    return boxes_to_draw, texts


def draw_ocr_boxes(img, boxes):
    out = img.copy()
    for box, text, conf in boxes:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts[0][0]
        cv2.putText(out, f"{text} {int(conf*100)}%", (x, max(y - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def find_and_draw_contours(frame):
    """Show ALL contours + highlight plate-shaped ones."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

    out = frame.copy()
    plate_crops = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:   # skip tiny noise
            continue
        p    = cv2.arcLength(c, True)
        a    = cv2.approxPolyDP(c, 0.02 * p, True)
        x, y, w, h = cv2.boundingRect(a)
        aspect = w / max(h, 1)

        # Plate-like: 1.8 < aspect < 6, wide enough
        if 1.8 < aspect < 6.5 and w > 60 and h > 15:
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(out, f"{w}x{h} a={aspect:.1f}", (x, max(y-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            crop = frame[y:y+h, x:x+w]
            if h < 80:
                scale = 80 / h
                crop  = cv2.resize(crop, (int(w*scale), 80), interpolation=cv2.INTER_CUBIC)
            plate_crops.append((x, y, w, h, crop))
        else:
            # draw non-plate contours dimly so you can see what's being rejected
            cv2.rectangle(out, (x, y), (x+w, y+h), (40, 40, 120), 1)

    print(f"[CONTOUR] Total kept: {len(cnts)}  Plate-shaped: {len(plate_crops)}")
    for i, (x,y,w,h,_) in enumerate(plate_crops):
        print(f"  Region {i}: x={x} y={y} w={w} h={h} aspect={w/max(h,1):.2f}")

    return out, plate_crops, edged


frame_count = 0
last_ocr_time = 0
last_ocr_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[DEBUG] Cannot read frame!")
        time.sleep(0.5)
        continue

    frame_count += 1
    disp = frame.copy()

    # Every 30 frames: run contour detection and auto-OCR plate regions
    if frame_count % 30 == 0:
        contour_vis, plate_crops, edged = find_and_draw_contours(frame)

        all_boxes = []
        if plate_crops:
            print(f"[AUTO] Found {len(plate_crops)} plate region(s), running OCR...")
            for i, (x, y, w, h, crop) in enumerate(plate_crops):
                # Try multiple preprocessed versions
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                enhanced = cv2.cvtColor(clahe.apply(gray_crop), cv2.COLOR_GRAY2BGR)
                versions = [crop, enhanced]
                for j, v in enumerate(versions):
                    boxes, texts = run_ocr_full(v, label=f"region{i}_v{j}", conf_thresh=0.2)
                    all_boxes.extend(boxes)
        else:
            print("[AUTO] No plate-shaped regions found — showing contour view")
            print("       TIP: The plate might be too small, or lighting is poor.")
            print("       Press 'f' to force OCR on full frame.")

        last_ocr_boxes = all_boxes
        last_ocr_time  = time.time()

    # Draw contour outlines on display
    contour_vis_disp, _, _ = find_and_draw_contours(frame)
    disp = contour_vis_disp

    # Draw OCR boxes from last run
    disp = draw_ocr_boxes(disp, last_ocr_boxes)

    # Status overlay
    elapsed = time.time() - last_ocr_time
    h_frame, w_frame = disp.shape[:2]
    cv2.putText(disp, f"Frame:{frame_count}  Last OCR: {elapsed:.1f}s ago",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(disp, "s=save  f=force OCR  q=quit",
                (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("PLATE DEBUG", disp)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        fname = f"debug_frame_{frame_count}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[DEBUG] Saved raw frame to {fname}")

    elif key == ord('f'):
        print("\n[FORCE OCR] Running OCR on FULL frame (no contour filtering)...")

        # Also try grayscale and enhanced
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced  = cv2.cvtColor(clahe.apply(gray_full), cv2.COLOR_GRAY2BGR)
        upscaled  = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2),
                               interpolation=cv2.INTER_CUBIC)

        all_boxes = []
        for label, img in [("original", frame), ("enhanced", enhanced), ("2x", upscaled)]:
            boxes, _ = run_ocr_full(img, label=label, conf_thresh=0.15)
            all_boxes.extend(boxes)

        last_ocr_boxes = all_boxes
        last_ocr_time  = time.time()

cap.release()
cv2.destroyAllWindows()
