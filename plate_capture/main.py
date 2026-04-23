# main.py  — PaddleOCR 2.7.3  (multi-pipeline + cross-validation + vote-to-confirm)
import os, logging
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
logging.disable(logging.WARNING)

import cv2, threading, time, queue, re, configparser, socket, json, base64
from datetime import datetime
from io import BytesIO
from collections import Counter
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from plate_saver import PlateSaver

# ── CONFIG ────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("config.ini")

FRAME_WIDTH      = int(config.get("General", "camera_width"))
FRAME_HEIGHT     = int(config.get("General", "camera_height"))

entry_camera_url = config.get("Cameras", "entry_camera_url")
try:    entry_camera_url = int(entry_camera_url)
except ValueError: pass

exit_camera_url = config.get("Cameras", "exit_camera_url")
try:    exit_camera_url = int(exit_camera_url)
except ValueError: pass

SERVER_HOST      = config.get("Server", "host")
SERVER_PORT      = int(config.get("Server", "port"))
CAPTURES_DIR     = config.get("Storage", "captures_dir")
COOLDOWN_SECONDS = int(config.get("Storage", "cooldown_seconds"))
MOTION_THRESHOLD = int(config.get("Detection", "motion_threshold"))
PROCESS_INTERVAL = int(config.get("Detection", "process_interval"))
REQUIRED_HITS    = int(config.get("Detection", "required_hits"))

OCR_INTERVAL_SECONDS = 3   # scan every N seconds regardless of motion

# Minimum number of preprocessing pipelines that must agree on the SAME plate
# before it is treated as a confirmed candidate.
PIPELINE_AGREEMENT_MIN = 2

# ── INIT ──────────────────────────────────────────────────────────────────────
ocr_engine        = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
plate_saver       = PlateSaver(captures_dir=CAPTURES_DIR, cooldown_seconds=COOLDOWN_SECONDS)
connected_clients = []
detection_lock    = threading.Lock()

# ── VALID INDIAN STATE / UT CODES ─────────────────────────────────────────────
VALID_STATES = {
    'AN','AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ',
    'HP','HR','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP',
    'MZ','NL','OD','OR','PB','PY','RJ','SK','TG','TN','TR','TS',
    'UP','UT','WB',
}

def snap_state_code(s2: str) -> str | None:
    if s2 in VALID_STATES:
        return s2
    ocr_confusions = {
        '0': ['O', 'D', 'Q'],
        'O': ['0', 'D'],
        '1': ['I', 'L'],
        'I': ['1', 'L'],
        '5': ['S'],
        'S': ['5'],
        '8': ['B'],
        'B': ['8'],
        '2': ['Z'],
        'Z': ['2'],
        'D': ['0', 'O'],
    }
    for pos in range(2):
        for replacement in ocr_confusions.get(s2[pos], []):
            candidate = (replacement + s2[1]) if pos == 0 else (s2[0] + replacement)
            if candidate in VALID_STATES:
                return candidate
    return None

# ── TCP SERVER ────────────────────────────────────────────────────────────────
def start_tcp_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"[SERVER] Listening on {host}:{port}")
    while True:
        cs, addr = server.accept()
        threading.Thread(target=handle_client, args=(cs, addr), daemon=True).start()

def handle_client(cs, addr):
    try:
        cs.settimeout(10)
        if cs.recv(1024).decode().strip() == "GET":
            with detection_lock:
                connected_clients.append((cs, addr))
            print(f"[SERVER] Client {addr} registered.")
        else:
            cs.close()
    except Exception as e:
        print(f"[SERVER] {e}")
        cs.close()

def broadcast(message: str):
    with detection_lock:
        dead = []
        for cs, addr in connected_clients:
            try:   cs.sendall((message + "\n").encode())
            except: dead.append((cs, addr))
        for c in dead:
            connected_clients.remove(c)

# ── IMAGE PREPROCESSING PIPELINES ─────────────────────────────────────────────
def build_pipelines(frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Return a list of (pipeline_name, processed_image) for OCR.
    Each pipeline highlights plate characters differently — combining their
    results through cross-validation dramatically improves accuracy.
    """
    pipelines = []

    # 1. Original colour frame
    pipelines.append(("original", frame))

    # 2. Grayscale only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pipelines.append(("gray", gray_bgr))

    # 3. CLAHE (contrast-limited adaptive histogram equalisation) — best for
    #    low-contrast or overexposed plates
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    pipelines.append(("clahe", cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)))

    # 4. Otsu global threshold — black & white, removes background clutter
    _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pipelines.append(("bw_otsu", cv2.cvtColor(bw_otsu, cv2.COLOR_GRAY2BGR)))

    # 5. Inverted Otsu — for dark-on-light plates vs light-on-dark plates
    pipelines.append(("bw_inv", cv2.cvtColor(cv2.bitwise_not(bw_otsu), cv2.COLOR_GRAY2BGR)))

    # 6. Adaptive threshold — handles uneven lighting (shadows / reflections)
    adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    pipelines.append(("adaptive", cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)))

    # 7. Sharpened — helps with blurry/motion-blurred plates
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(frame, -1, kernel)
    pipelines.append(("sharp", sharp))

    return pipelines


# ── OCR ───────────────────────────────────────────────────────────────────────
def run_ocr(img_bgr: np.ndarray, conf_threshold: float = 0.3) -> tuple:
    try:
        result = ocr_engine.ocr(img_bgr, cls=True)
        if not result or result[0] is None:
            return "", []
        lines, boxes = [], []
        for line in result[0]:
            text = line[1][0].upper()
            conf = line[1][1]
            if conf > conf_threshold:
                lines.append(text)
                boxes.append((line[0], text, conf))
        return " ".join(lines), boxes
    except Exception as e:
        print(f"[OCR] {e}")
        return "", []


def ocr_full_frame(frame: np.ndarray) -> tuple:
    """
    Run OCR across ALL preprocessing pipelines.
    Returns:
        per_pipeline_texts : list of non-empty text strings (one per pipeline)
        all_boxes          : combined bounding boxes from all pipelines
        pipeline_log       : list of (pipeline_name, text) for debug printing
    """
    pipelines   = build_pipelines(frame)
    per_pipeline_texts = []
    all_boxes   = []
    pipeline_log = []

    for name, img in pipelines:
        text, boxes = run_ocr(img)
        pipeline_log.append((name, text))
        if text:
            per_pipeline_texts.append(text)
            all_boxes.extend(boxes)

    return per_pipeline_texts, all_boxes, pipeline_log


# ── PLATE MATCHING ────────────────────────────────────────────────────────────
_STD_LOOSE  = re.compile(r'[A-Z0-9]{2}\d{0,2}[A-Z0-9]{1,3}\d{3,4}')
_BH_LOOSE   = re.compile(r'\d{2}BH\d{3,4}[A-Z]{1,2}')
_STD_STRICT = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')
_BH_STRICT  = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')

_D2L = {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G'}
_L2D = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0','D':'0'}

def _fl(ch): return _D2L.get(ch, ch)
def _fd(ch): return _L2D.get(ch, ch)

def normalize_plate(raw: str) -> str | None:
    p = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(p) < 8:
        return None
    t = list(p)
    n = len(t)
    is_bharat = n >= 4 and t[0].isdigit() and t[1].isdigit() and ''.join(t[2:4]) == 'BH'
    if is_bharat:
        t[0] = _fd(t[0]); t[1] = _fd(t[1])
        for i in range(4, min(8, n)): t[i] = _fd(t[i])
        for i in range(8, n):         t[i] = _fl(t[i])
    else:
        t[0] = _fl(t[0]); t[1] = _fl(t[1])
        t[2] = _fd(t[2]); t[3] = _fd(t[3])
        for i in range(4, max(4, n - 4)): t[i] = _fl(t[i])
        for i in range(max(0, n - 4), n): t[i] = _fd(t[i])
        state   = ''.join(t[:2])
        snapped = snap_state_code(state)
        if snapped is None:
            return None
        t[0] = snapped[0]; t[1] = snapped[1]
    return ''.join(t)

def pad_to_4_digits(plate: str) -> str:
    m = re.match(r'^([A-Z]{2}\d{2}[A-Z]{1,3})(\d{3})$', plate)
    return m.group(1) + m.group(2) + '0' if m else plate

_NOISE = {
    'IND','INDIA','TATA','MARUTI','SUZUKI','HYUNDAI','MAHINDRA','HONDA',
    'TOYOTA','FORD','NISSAN','KIA','MG','RENAULT','BAJAJ','HERO','TVS',
    'ROYAL','ENFIELD','YAMAHA','KAWASAKI','ASHOK','LEYLAND','EICHER',
    'FORCE','ISUZU','VOLVO','SCANIA','BUS','TRUCK','AUTO','GOOD','INO',
}

def is_valid_plate_strict(text: str) -> bool:
    if not text:
        return False
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not (6 <= len(text) <= 10):
        return False
    if len(text) >= 8:
        if text[:2].isdigit() and text[2:4] == "BH":
            i = 4
            digit_count = 0
            while i < len(text) and text[i].isdigit() and digit_count < 4:
                digit_count += 1; i += 1
            if digit_count < 3:
                return False
            suffix = text[i:]
            return 1 <= len(suffix) <= 2 and suffix.isalpha()
    if not text[:2].isalpha():
        return False
    i = 2
    digit_count = 0
    while i < len(text) and text[i].isdigit() and digit_count < 2:
        digit_count += 1; i += 1
    if digit_count == 0:
        return False
    letter_count = 0
    while i < len(text) and text[i].isalpha() and letter_count < 2:
        letter_count += 1; i += 1
    remaining = text[i:]
    return 1 <= len(remaining) <= 4 and remaining.isdigit()

def extract_plates(pass_texts) -> list:
    if isinstance(pass_texts, str):
        pass_texts = [pass_texts]
    results = {}

    def _try(text: str):
        tokens = [re.sub(r'[^A-Z0-9]', '', t) for t in re.split(r'[\s\n\r]+', text.upper())]
        tokens = [t for t in tokens if t and t not in _NOISE and len(t) <= 12]
        if not tokens:
            return
        candidates = set()
        for w in range(1, 7):
            for i in range(len(tokens) - w + 1):
                candidates.add(''.join(tokens[i:i + w]))
        for cand in candidates:
            for m in _STD_LOOSE.findall(cand):
                norm = normalize_plate(m)
                if norm is None: continue
                norm = pad_to_4_digits(norm)
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm
            for m in _BH_LOOSE.findall(cand):
                norm = normalize_plate(m)
                if norm is None: continue
                bm = re.match(r'^(\d{2}BH)(\d{3})([A-Z]{1,2})$', norm)
                if bm: norm = bm.group(1) + bm.group(2) + '0' + bm.group(3)
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm

    for text in pass_texts:
        _try(text)
    _try(" ".join(pass_texts))
    return list(results.values())


# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────
def make_digit_core(plate: str) -> str:
    """
    Stable core = last 4 digits of the plate number.
    OR051234  → '1234'
    OR05AB1234 → '1234'
    Both map to same core, preventing double-save.
    """
    digits = re.sub(r'[^0-9]', '', plate)
    return digits[-4:] if len(digits) >= 4 else digits

def cross_validate_plates(per_pipeline_texts: list[str]) -> dict[str, list]:
    """
    Run extract_plates on every pipeline's text independently.
    Group results by digit-core.
    Return: { digit_core: [list of plate strings found across pipelines] }
    """
    core_map: dict[str, list] = {}
    for text in per_pipeline_texts:
        plates = extract_plates([text])
        for plate in plates:
            core = make_digit_core(plate)
            if core not in core_map:
                core_map[core] = []
            core_map[core].append(plate)
    return core_map


def pick_best_plate(plate_list: list[str]) -> str:
    """
    From a list of plate strings (possibly from different pipelines / frames),
    pick the best one using these priorities (in order):
      1. Longest plate (most complete)
      2. Most frequent among the list
      3. Lexicographic tie-break
    """
    if not plate_list:
        return ""
    counts = Counter(plate_list)
    return max(counts.keys(), key=lambda p: (len(p), counts[p]))


# ── DISPLAY ───────────────────────────────────────────────────────────────────
def draw_ocr_boxes(disp, boxes):
    for box, text, conf in boxes:
        pts   = np.array(box, np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
        cv2.polylines(disp, [pts], True, color, 2)
        x, y = pts[0][0]
        cv2.putText(disp, text, (x, max(y - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return disp

def draw_panel(disp, hits, required_hits, raw_text, best_plate, next_ocr_in, fps, cam, pipeline_summary=""):
    h, w = disp.shape[:2]
    pw   = 360
    ov   = disp.copy()
    cv2.rectangle(ov, (w - pw, 0), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.75, disp, 0.25, 0, disp)
    cv2.line(disp, (w - pw, 0), (w - pw, h), (0, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28

    def put(txt, col=(210, 210, 210), sc=0.5, th=1):
        nonlocal y
        cv2.putText(disp, txt, (w - pw + 8, y), font, sc, col, th)
        y += int(sc * 38 + 4)

    put(f"{cam}  FPS:{fps:.1f}", (0, 220, 220), 0.55, 2)
    ocr_col = (0, 255, 80) if next_ocr_in < 0.3 else (0, 200, 255)
    put(f"Next scan: {next_ocr_in:.1f}s", ocr_col)

    if best_plate:
        put(f"PLATE: {best_plate}", (0, 255, 100), 0.65, 2)

    y += 4
    put("CANDIDATES:", (170, 170, 170), 0.44)
    if hits:
        for plate, cnt in sorted(hits.items(), key=lambda x: -x[1])[:6]:
            bar = "█" * min(cnt, required_hits) + "░" * max(0, required_hits - cnt)
            col = (0, 255, 80) if cnt >= required_hits else (0, 165, 255)
            put(f"{plate}  {bar} {cnt}/{required_hits}", col, 0.45, 1)
    else:
        put("[none]", (80, 80, 80), 0.44)

    y += 4
    put("PIPELINES:", (170, 170, 170), 0.44)
    if pipeline_summary:
        for line in pipeline_summary.split('\n')[:6]:
            put(line[:42], (100, 180, 100), 0.38)
            if y > h - 16: break

    y += 4
    put("OCR:", (170, 170, 170), 0.44)
    short = raw_text[:90] + ("…" if len(raw_text) > 90 else "") or "[none]"
    for i in range(0, len(short), 36):
        put(short[i:i+36], (110, 110, 110), 0.38)
        if y > h - 16: break
    return disp


# ── DETECTION WORKER ──────────────────────────────────────────────────────────
def detect_worker(fq: queue.Queue, camera_name: str, shared: dict):
    """
    Detection logic:
    ─────────────────────────────────────────────────────────
    For each frame:
      1. Run OCR on 7 preprocessing pipelines (original, gray, CLAHE,
         Otsu B&W, inverted B&W, adaptive threshold, sharpened).
      2. Extract plates from each pipeline independently.
      3. Cross-validate: group by last-4-digit core.
      4. Only accept a core if ≥ PIPELINE_AGREEMENT_MIN pipelines found it.
      5. Among agreed plates, pick the longest/most-voted variant as `best`.
      6. Accumulate frame-level hits for each core.
      7. Once hits ≥ REQUIRED_HITS AND length is ≥ 8, save the plate.
    ─────────────────────────────────────────────────────────
    """
    # digit_core → list of plate strings seen across ALL frames
    core_history:  dict[str, list] = {}
    # digit_core → hit count (frames)
    core_hits:     dict[str, int]  = {}
    # digit_core → frame count since last seen (for stale cleanup)
    core_last_seen:dict[str, int]  = {}
    # cores saved this session
    saved_cores:   set[str]        = set()
    frame_idx = 0

    while True:
        try:
            item = fq.get(timeout=5)
        except queue.Empty:
            continue
        if item is None:
            break

        frame, cap_time = item
        frame_idx += 1

        # ── 1. Multi-pipeline OCR ──────────────────────────────────────────
        per_pipeline_texts, all_boxes, pipeline_log = ocr_full_frame(frame)

        all_text = " ".join(per_pipeline_texts)
        shared['last_raw_text'] = all_text
        shared['last_boxes']    = all_boxes

        # Build pipeline summary for display
        pl_summary_lines = []
        for pname, ptext in pipeline_log:
            snippet = ptext[:28] + "…" if len(ptext) > 28 else ptext
            pl_summary_lines.append(f"{pname[:8]}: {snippet or '-'}")
        shared['pipeline_summary'] = "\n".join(pl_summary_lines)

        if all_text:
            print(f"[{camera_name}][OCR] Pipelines used: {len(per_pipeline_texts)}/{len(pipeline_log)}")
            for pname, ptext in pipeline_log:
                if ptext:
                    print(f"  [{pname}] {ptext[:80]}")

        # ── 2. Cross-validate across pipelines ────────────────────────────
        core_map = cross_validate_plates(per_pipeline_texts)

        # ── 3. Filter: core must be agreed by ≥ PIPELINE_AGREEMENT_MIN ────
        agreed_cores = {
            core: plates
            for core, plates in core_map.items()
            if len(plates) >= PIPELINE_AGREEMENT_MIN
        }

        if not agreed_cores:
            # Also try reading ALL pipeline texts combined as a fallback
            all_plates = extract_plates(per_pipeline_texts)
            if all_plates:
                for plate in all_plates:
                    core = make_digit_core(plate)
                    agreed_cores[core] = agreed_cores.get(core, []) + [plate]
                print(f"[{camera_name}] ⚠️  No pipeline agreement — using combined fallback")
            else:
                print(f"[{camera_name}] No valid plates found in any pipeline.")
                shared['best_plate'] = None

        # Update hit tracking display
        shared['hits'] = {
            pick_best_plate(core_history.get(core, [core])): core_hits.get(core, 0)
            for core in core_hits
        }

        seen_cores_this_frame = set()

        for core, plate_list in agreed_cores.items():
            # Accumulate history across frames
            if core not in core_history:
                core_history[core] = []
            core_history[core].extend(plate_list)

            # One hit per frame per core
            if core not in seen_cores_this_frame:
                # If a longer plate appears than what we've seen before,
                # slightly reset hits to avoid locking in a short plate too early
                current_best = pick_best_plate(core_history[core])
                prev_best    = pick_best_plate(core_history[core][:-len(plate_list)] or [current_best])
                if len(current_best) > len(prev_best) and core_hits.get(core, 0) >= REQUIRED_HITS - 1:
                    print(f"[{camera_name}] 🔄 Longer plate emerged ({current_best} > {prev_best}), extending window")
                    core_hits[core] = max(1, (core_hits.get(core, 0) // 2))

                core_hits[core]      = core_hits.get(core, 0) + 1
                core_last_seen[core] = frame_idx
                seen_cores_this_frame.add(core)

            best  = pick_best_plate(core_history[core])
            hits  = core_hits[core]
            agreement_count = len(plate_list)

            shared['best_plate'] = best
            print(f"[{camera_name}] Core:{core}  best={best} (len={len(best)})  "
                  f"hits={hits}/{REQUIRED_HITS}  pipelines_agree={agreement_count}")

            # ── SAVE GATE ──────────────────────────────────────────────────
            if hits < REQUIRED_HITS:
                continue

            # Must be a complete plate (≥ 8 chars for standard, ≥ 9 for BH)
            if len(best) < 8:
                print(f"[{camera_name}] ⏳ Plate too short ({best}), waiting for full read…")
                continue

            if core in saved_cores:
                print(f"[{camera_name}] ⏳ Already saved core {core} this session")
                core_hits[core] = 0
                continue

            # ── FINAL CROSS-CHECK: re-read plate region isolated ──────────
            # Crop the detected region and run one more dedicated OCR pass
            boxes      = shared.get('last_boxes', [])
            save_frame = frame
            if boxes:
                h_img, w_img = frame.shape[:2]
                all_pts = []
                for box, text, conf in boxes:
                    all_pts.extend(box)
                all_pts = np.array(all_pts, dtype=np.int32)
                x1, y1 = all_pts[:, 0].min(), all_pts[:, 1].min()
                x2, y2 = all_pts[:, 0].max(), all_pts[:, 1].max()
                pad_x = int((x2 - x1) * 0.4)
                pad_y = int((y2 - y1) * 0.4)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w_img, x2 + pad_x)
                y2 = min(h_img, y2 + pad_y)
                save_frame = frame[y1:y2, x1:x2]

                # Final verification OCR on the cropped plate region only
                if save_frame.size > 0:
                    verify_texts, _, _ = ocr_full_frame(save_frame)
                    verify_core_map    = cross_validate_plates(verify_texts)
                    verify_match       = core in verify_core_map
                    if not verify_match:
                        # Try any overlapping core in verify result
                        for vc, vplates in verify_core_map.items():
                            if vc == core or best.endswith(vc) or vc in best:
                                verify_match = True
                                break
                    if not verify_match:
                        print(f"[{camera_name}] ❌ Final verify FAILED for {best} — skipping save")
                        # Don't hard-reject; let it accumulate more hits
                        core_hits[core] = max(0, core_hits[core] - 2)
                        continue
                    else:
                        print(f"[{camera_name}] ✅ Final verify PASSED for {best}")

            # ── SAVE ──────────────────────────────────────────────────────
            saved = plate_saver.save(frame=save_frame, plate_number=best,
                                     camera_name=camera_name)
            if saved:
                print(f"[{camera_name}] ✅ SAVED: {best}  →  {saved}")
                saved_cores.add(core)
                img = Image.fromarray(cv2.cvtColor(save_frame, cv2.COLOR_BGR2RGB))
                buf = BytesIO()
                img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                broadcast(json.dumps({
                    "camera_name": camera_name.lower(),
                    "detected_text": best,
                    "image": b64,
                    "time": cap_time,
                    "saved_path": saved
                }))
            else:
                print(f"[{camera_name}] ⏳ plate_saver cooldown: {best}")

            core_hits[core] = 0

        # ── STALE CORE CLEANUP (not seen in last 10 frames) ───────────────
        stale = [
            c for c, last in core_last_seen.items()
            if (frame_idx - last) > 10 and c not in saved_cores
        ]
        for c in stale:
            core_history.pop(c, None)
            core_hits.pop(c, None)
            core_last_seen.pop(c, None)


# ── CAMERA WORKER ─────────────────────────────────────────────────────────────
def camera_worker(url, name: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"[{name}] Camera opened.")

    fq            = queue.Queue(maxsize=2)
    frame_count   = 0
    prev_time     = time.time()
    fps           = 0.0
    last_ocr_time = 0.0

    shared = {
        'last_raw_text':   "",
        'last_boxes':      [],
        'last_plates':     [],
        'best_plate':      None,
        'hits':            {},
        'pipeline_summary': "",
    }

    threading.Thread(target=detect_worker, args=(fq, name, shared), daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] Cannot read frame, retrying…")
                time.sleep(0.5)
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(url)
                continue

            frame_count += 1
            if frame_count % 15 == 0:
                now = time.time()
                fps = 15 / max(now - prev_time, 0.001)
                prev_time = now

            now = time.time()
            if (now - last_ocr_time) >= OCR_INTERVAL_SECONDS and fq.qsize() < 1:
                fq.put((frame.copy(), datetime.now().isoformat()))
                last_ocr_time = now

            disp = frame.copy()
            disp = draw_ocr_boxes(disp, shared.get('last_boxes', []))
            disp = draw_panel(
                disp,
                hits             = shared.get('hits', {}),
                required_hits    = REQUIRED_HITS,
                raw_text         = shared.get('last_raw_text', ""),
                best_plate       = shared.get('best_plate'),
                next_ocr_in      = max(0.0, OCR_INTERVAL_SECONDS - (time.time() - last_ocr_time)),
                fps              = fps,
                cam              = name,
                pipeline_summary = shared.get('pipeline_summary', ""),
            )
            cv2.imshow(name, disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        fq.put(None)
        cap.release()
        cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    threading.Thread(target=start_tcp_server,
                     args=(SERVER_HOST, SERVER_PORT), daemon=True).start()
    t = threading.Thread(target=camera_worker,
                         args=(entry_camera_url, "Entry"), daemon=True)
    t.start()
    print("[INFO] Press 'q' in camera window to quit.")
    try:
        while t.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()