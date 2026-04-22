# main.py  — PaddleOCR 2.7.3  (state-code validated, vote-to-confirm)
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
CONFIG_PATH = os.path.join(APP_DIR, "config.ini")
config.read(CONFIG_PATH)

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
FORCED_STATE_CODE = config.get("Plate", "forced_state_code", fallback="").strip().upper()

OCR_INTERVAL_SECONDS = 3   # scan every N seconds regardless of motion
DISPLAY_WINDOWS = os.environ.get(
    "PLATE_CAPTURE_GUI",
    "1",
) == "1"
OCR_MAX_SIDE = 1280
MAX_PLATE_CROPS = 3
FULL_FRAME_FALLBACK_EVERY = 4

if not os.path.isabs(CAPTURES_DIR):
    CAPTURES_DIR = os.path.join(RUN_DIR, CAPTURES_DIR)

# ── INIT ──────────────────────────────────────────────────────────────────────
def create_ocr_engine():
    try:
        return PaddleOCR(use_textline_orientation=True, lang='en')
    except (TypeError, ValueError):
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

ocr_engine        = create_ocr_engine()
plate_saver       = PlateSaver(captures_dir=CAPTURES_DIR, cooldown_seconds=COOLDOWN_SECONDS)
connected_clients = []
detection_lock    = threading.Lock()
display_lock      = threading.Lock()
display_frames: dict[str, np.ndarray] = {}
stop_event        = threading.Event()

def open_camera_capture(source):
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        elif sys.platform == "darwin":
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)

def parse_ocr_result(result, conf_threshold: float = 0.3) -> tuple[str, list]:
    if not result:
        return "", []

    item = result[0]
    rec_texts = item.get("rec_texts", []) if hasattr(item, "get") else []
    rec_scores = item.get("rec_scores", []) if hasattr(item, "get") else []
    rec_polys = item.get("rec_polys") if hasattr(item, "get") else None
    dt_polys = item.get("dt_polys") if hasattr(item, "get") else None
    polys = rec_polys or dt_polys or []

    lines, boxes = [], []
    for text, conf, poly in zip(rec_texts, rec_scores, polys):
        if conf is None or conf <= conf_threshold:
            continue
        norm_text = str(text).upper()
        points = np.array(poly).astype(int).tolist()
        lines.append(norm_text)
        boxes.append((points, norm_text, float(conf)))
    return " ".join(lines), boxes

# ── VALID INDIAN STATE / UT CODES ─────────────────────────────────────────────
# Used to snap OCR noise like 'OO' → 'OD', 'FA' → reject, etc.
VALID_STATES = {
    'AN','AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ',
    'HP','HR','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP',
    'MZ','NL','OD','OR','PB','PY','RJ','SK','TG','TN','TR','TS',
    'UP','UT','WB',
}

def snap_state_code(s2: str) -> str | None:
    """
    Return the state code as-is if valid.
    If not valid, try single-character substitution fixes and return best match.
    Returns None if no valid state found.
    """
    if s2 in VALID_STATES:
        return s2
    # Try fixing each position using OCR confusion map
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
    return None   # Not a valid state code at all — reject this plate


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

# ── OCR ───────────────────────────────────────────────────────────────────────
def resize_for_ocr(img_bgr: np.ndarray, max_side: int = OCR_MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_bgr, 1.0

    scale = max_side / float(longest)
    resized = cv2.resize(
        img_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale

def scale_boxes(boxes: list, scale: float, offset_x: int = 0, offset_y: int = 0) -> list:
    if scale == 0:
        return boxes

    inv = 1.0 / scale
    scaled = []
    for box, text, conf in boxes:
        pts = []
        for x, y in box:
            pts.append([int(x * inv + offset_x), int(y * inv + offset_y)])
        scaled.append((pts, text, conf))
    return scaled

def find_plate_regions(frame: np.ndarray, max_regions: int = MAX_PLATE_CROPS) -> list[tuple[int, int, int, int]]:
    resized, scale = resize_for_ocr(frame, max_side=960)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blur, 60, 180)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    inv = 1.0 / scale

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if not (2.0 <= aspect <= 6.5):
            continue
        if w < 50 or h < 16:
            continue

        x1 = max(0, int((x - w * 0.15) * inv))
        y1 = max(0, int((y - h * 0.25) * inv))
        x2 = min(frame.shape[1], int((x + w * 1.15) * inv))
        y2 = min(frame.shape[0], int((y + h * 1.25) * inv))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((area, (x1, y1, x2, y2)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    regions = []
    for _, region in candidates:
        if region not in regions:
            regions.append(region)
        if len(regions) >= max_regions:
            break
    return regions

def run_ocr(img_bgr: np.ndarray, conf_threshold: float = 0.3) -> tuple:
    try:
        resized, scale = resize_for_ocr(img_bgr)
        try:
            result = ocr_engine.predict(resized, use_textline_orientation=True)
            text, boxes = parse_ocr_result(result, conf_threshold)
            return text, scale_boxes(boxes, scale)
        except TypeError:
            result = ocr_engine.ocr(resized, cls=True)
            if not result or result[0] is None:
                return "", []
            lines, boxes = [], []
            for line in result[0]:
                text = line[1][0].upper()
                conf = line[1][1]
                if conf > conf_threshold:
                    lines.append(text)
                    boxes.append((line[0], text, conf))
            return " ".join(lines), scale_boxes(boxes, scale)
    except Exception as e:
        print(f"[OCR] {e}")
        return "", []

def ocr_full_frame(frame: np.ndarray, full_frame_fallback: bool = False) -> tuple:
    per_pass, all_boxes = [], []

    for x1, y1, x2, y2 in find_plate_regions(frame):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        text, boxes = run_ocr(crop)
        if text:
            per_pass.append(text)
            all_boxes.extend(scale_boxes(boxes, 1.0, offset_x=x1, offset_y=y1))

    if per_pass:
        return per_pass, all_boxes

    if full_frame_fallback or not per_pass:
        resized, scale = resize_for_ocr(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        for img in [resized, enhanced]:
            text, boxes = run_ocr(img)
            if text:
                per_pass.append(text)
                all_boxes.extend(scale_boxes(boxes, scale))
    return per_pass, all_boxes

# ── PLATE MATCHING ────────────────────────────────────────────────────────────
_STD_LOOSE  = re.compile(r'[A-Z0-9]{2}\d{2}[A-Z0-9]{0,3}\d{1,4}')
_BH_LOOSE   = re.compile(r'\d{2}BH[A-Z0-9]{2,6}')

_D2L = {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G'}
_L2D = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0','D':'0'}

def _fl(ch): return _D2L.get(ch, ch)   # force-letter
def _fd(ch): return _L2D.get(ch, ch)   # force-digit

def _looks_like_letters(chars: list[str]) -> bool:
    return all(_fl(ch).isalpha() for ch in chars)

def _looks_like_digits(chars: list[str]) -> bool:
    return all(_fd(ch).isdigit() for ch in chars)

def normalize_plate(raw: str) -> str | None:
    """
    Normalize OCR noise using position rules.
    Also validates state code — returns None if state is not a known Indian code.
    """
    p = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(p) < 6:
        return None

    t = list(p)
    n = len(t)

    bh_prefix = ''.join([_fd(t[0]), _fd(t[1]), _fl(t[2]), _fl(t[3])]) if n >= 4 else ""
    is_bharat = n >= 6 and bh_prefix[:2].isdigit() and bh_prefix[2:4] == 'BH'

    if is_bharat:
        t[0] = _fd(t[0]); t[1] = _fd(t[1])
        t[2] = 'B'; t[3] = 'H'

        for suffix_len in (2, 1):
            digit_end = n - suffix_len
            if digit_end <= 4:
                continue
            digit_part = t[4:digit_end]
            suffix_part = t[digit_end:]
            if 1 <= len(digit_part) <= 4 and 1 <= len(suffix_part) <= 2:
                if _looks_like_digits(digit_part) and _looks_like_letters(suffix_part):
                    for i in range(4, digit_end):
                        t[i] = _fd(t[i])
                    for i in range(digit_end, n):
                        t[i] = _fl(t[i])
                    return ''.join(t)
        return None
    else:
        if n < 5 or n > 11:
            return None

        # State (pos 0,1) → letters
        t[0] = _fl(t[0]); t[1] = _fl(t[1])
        # District (pos 2,3) → digits
        t[2] = _fd(t[2]); t[3] = _fd(t[3])

        matched = False
        for num_len in range(4, 0, -1):
            num_start = n - num_len
            if num_start < 4:
                continue
            series_part = t[4:num_start]
            number_part = t[num_start:]
            if len(series_part) > 3:
                continue
            if _looks_like_letters(series_part) and _looks_like_digits(number_part):
                for i in range(4, num_start):
                    t[i] = _fl(t[i])
                for i in range(num_start, n):
                    t[i] = _fd(t[i])
                matched = True
                break

        if not matched:
            return None

        # ── STATE CODE VALIDATION ─────────────────────────────────────────
        if FORCED_STATE_CODE:
            if len(FORCED_STATE_CODE) != 2 or FORCED_STATE_CODE not in VALID_STATES:
                return None
            t[0] = FORCED_STATE_CODE[0]
            t[1] = FORCED_STATE_CODE[1]
        else:
            state = ''.join(t[:2])
            snapped = snap_state_code(state)
            if snapped is None:
                return None          # Not a real Indian state — discard plate
            t[0] = snapped[0]
            t[1] = snapped[1]

    return ''.join(t)

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

    # Length constraint
    if not (6 <= len(text) <= 11):
        return False

    # ── 1. BH SERIES CHECK (FIRST PRIORITY) ─────────────────────
    # Format: 22BH1AA ... 22BH1234AA
    if len(text) >= 6:
        if text[:2].isdigit() and text[2:4] == "BH":
            i = 4

            # Next must be digits (1–4)
            digit_count = 0
            while i < len(text) and text[i].isdigit() and digit_count < 4:
                digit_count += 1
                i += 1

            if digit_count < 1:
                return False

            # Last must be letters (1–2)
            suffix = text[i:]
            if 1 <= len(suffix) <= 2 and suffix.isalpha():
                return True

            return False

    # ── 2. STANDARD FORMAT (FALLBACK) ───────────────────────────

    # First 2 must be letters (state)
    if not text[:2].isalpha():
        return False

    i = 2

    # Next must be 1–2 digits (RTO)
    digit_count = 0
    while i < len(text) and text[i].isdigit() and digit_count < 2:
        digit_count += 1
        i += 1

    if digit_count == 0:
        return False

    # Optional letters (0–3)
    letter_count = 0
    while i < len(text) and text[i].isalpha() and letter_count < 3:
        letter_count += 1
        i += 1

    # Remaining must be digits (1–4)
    remaining = text[i:]
    if not (1 <= len(remaining) <= 4 and remaining.isdigit()):
        return False

    return True

def extract_plates(pass_texts) -> list:
    if isinstance(pass_texts, str):
        pass_texts = [pass_texts]

    results = {}   # core → plate string

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
                if norm is None:
                    continue
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm
            for m in _BH_LOOSE.findall(cand):
                norm = normalize_plate(m)
                if norm is None:
                    continue
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm

    for text in pass_texts:
        _try(text)
    _try(" ".join(pass_texts))

    return list(results.values())

def select_plate_crop(frame: np.ndarray, boxes: list, plate: str) -> np.ndarray:
    """
    Crop around OCR boxes that most likely belong to the confirmed plate.
    Falls back to the original frame if no matching OCR geometry is found.
    """
    if not boxes or not plate:
        return frame

    target_core = plate[-6:]
    matched_pts = []

    for box, text, conf in boxes:
        matches = extract_plates([text])
        if plate in matches or any(m[-6:] == target_core for m in matches):
            matched_pts.extend(box)

    if not matched_pts:
        return frame

    h_img, w_img = frame.shape[:2]
    pts = np.array(matched_pts, dtype=np.int32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()

    pad_x = max(12, int((x2 - x1) * 0.4))
    pad_y = max(12, int((y2 - y1) * 0.4))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]

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

def draw_panel(disp, hits, required_hits, raw_text, best_plate, next_ocr_in, fps, cam):
    h, w = disp.shape[:2]
    pw   = 310
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
    put("OCR:", (170, 170, 170), 0.44)
    short = raw_text[:90] + ("…" if len(raw_text) > 90 else "") or "[none]"
    for i in range(0, len(short), 36):
        put(short[i:i+36], (110, 110, 110), 0.38)
        if y > h - 16:
            break
    return disp

# ── DETECTION WORKER ──────────────────────────────────────────────────────────
def detect_worker(fq: queue.Queue, camera_name: str, shared: dict):
    # core → Counter of full plate strings (voting)
    core_votes: dict[str, Counter] = {}
    # core → hit count (how many frames this core appeared in)
    core_hits:  dict[str, int]     = {}
    ocr_runs = 0

    while True:
        try:
            item = fq.get(timeout=5)
        except queue.Empty:
            continue
        if item is None:
            break

        frame, cap_time = item

        ocr_runs += 1
        per_pass, all_boxes = ocr_full_frame(
            frame,
            full_frame_fallback=(ocr_runs % FULL_FRAME_FALLBACK_EVERY) == 0,
        )
        all_text = " ".join(per_pass)
        shared['last_raw_text'] = all_text
        shared['last_boxes']    = all_boxes

        plates = extract_plates(per_pass)
        shared['hits'] = {
            # Show best-voted plate per core in the UI
            next(iter(v.most_common(1)), (core, 0))[0]: core_hits.get(core, 0)
            for core, v in core_votes.items()
        }

        if all_text:
            print(f"[{camera_name}][OCR] {per_pass}")

        if not plates:
            if all_text:
                tokens = [re.sub(r'[^A-Z0-9]', '', t) for t in all_text.upper().split()]
                print(f"[{camera_name}] No match. Tokens: {[t for t in tokens if t]}")
            shared['best_plate'] = None
            continue

        seen_cores_this_frame = set()
        for plate in plates:
            core = plate[-6:]

            # Accumulate votes for this core
            if core not in core_votes:
                core_votes[core] = Counter()
            core_votes[core][plate] += 1

            # Count hits per core (one hit per frame, not per variant)
            if core not in seen_cores_this_frame:
                core_hits[core] = core_hits.get(core, 0) + 1
                seen_cores_this_frame.add(core)

            # Best plate = most-voted variant for this core
            best = core_votes[core].most_common(1)[0][0]
            shared['best_plate'] = best

            hits = core_hits[core]
            print(f"[{camera_name}] Core:{core}  best={best}  hits={hits}/{REQUIRED_HITS}")

            if hits >= REQUIRED_HITS:
                save_frame = select_plate_crop(frame, shared.get('last_boxes', []), best)

                saved = plate_saver.save(frame=save_frame, plate_number=best,
                                         camera_name=camera_name)
                if saved:
                    print(f"[{camera_name}] ✅ SAVED: {best}  →  {saved}")
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

                # Reset hit counter for this core
                core_hits[core] = 0

        # Drop idle cores so votes don't linger forever after a save/cooldown skip.
        stale = [c for c, h in core_hits.items() if h == 0]
        for c in stale:
            core_votes.pop(c, None)
            core_hits.pop(c, None)

# ── CAMERA WORKER ─────────────────────────────────────────────────────────────
def camera_worker(url, name: str):
    cap = open_camera_capture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"[{name}] Camera opened.")

    fq            = queue.Queue(maxsize=2)
    frame_count   = 0
    prev_time     = time.time()
    fps           = 0.0
    last_ocr_time = 0.0
    prev_gray     = None

    shared = {
        'last_raw_text': "",
        'last_boxes':    [],
        'last_plates':   [],
        'best_plate':    None,
        'hits':          {},
    }

    threading.Thread(target=detect_worker, args=(fq, name, shared), daemon=True).start()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] Cannot read frame, retrying…")
                time.sleep(0.5)
                cap.release()
                time.sleep(1)
                cap = open_camera_capture(url)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                continue

            frame_count += 1
            if frame_count % 15 == 0:
                now = time.time()
                fps = 15 / max(now - prev_time, 0.001)
                prev_time = now

            now = time.time()
            should_scan = False

            if PROCESS_INTERVAL <= 1 or frame_count % PROCESS_INTERVAL == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    should_scan = True
                elif MOTION_THRESHOLD <= 0:
                    should_scan = True
                else:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    motion_score = int(cv2.countNonZero(thresh))
                    should_scan = motion_score >= MOTION_THRESHOLD
                prev_gray = gray

            if (now - last_ocr_time) >= OCR_INTERVAL_SECONDS:
                should_scan = True

            if should_scan and fq.qsize() < 1:
                fq.put((frame.copy(), datetime.now().isoformat()))
                last_ocr_time = now

            disp = frame.copy()
            disp = draw_ocr_boxes(disp, shared.get('last_boxes', []))
            disp = draw_panel(
                disp,
                hits          = shared.get('hits', {}),
                required_hits = REQUIRED_HITS,
                raw_text      = shared.get('last_raw_text', ""),
                best_plate    = shared.get('best_plate'),
                next_ocr_in   = max(0.0, OCR_INTERVAL_SECONDS - (time.time() - last_ocr_time)),
                fps           = fps,
                cam           = name,
            )

            with display_lock:
                display_frames[name] = disp
    finally:
        try:
            fq.put(None, timeout=1)
        except queue.Full:
            pass
        cap.release()

def display_loop(camera_names: list[str]):
    while not stop_event.is_set():
        with display_lock:
            frames = [(name, display_frames.get(name)) for name in camera_names]

        for name, frame in frames:
            if frame is None:
                continue
            cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        time.sleep(0.01)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    threading.Thread(target=start_tcp_server,
                     args=(SERVER_HOST, SERVER_PORT), daemon=True).start()

    workers = []
    camera_sources = [("Entry", entry_camera_url)]
    if exit_camera_url != entry_camera_url:
        camera_sources.append(("Exit", exit_camera_url))

    for camera_name, camera_url in camera_sources:
        t = threading.Thread(target=camera_worker,
                             args=(camera_url, camera_name), daemon=True)
        t.start()
        workers.append(t)

    if DISPLAY_WINDOWS:
        print("[INFO] Press 'q' in camera window to quit.")
    else:
        print("[INFO] GUI windows disabled. Use Ctrl+C to stop.")
    try:
        if DISPLAY_WINDOWS:
            try:
                display_loop([camera_name for camera_name, _ in camera_sources])
            except cv2.error as e:
                print(f"[INFO] Display disabled: {e}")
                stop_event.set()
        else:
            while any(t.is_alive() for t in workers) and not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
        stop_event.set()
    finally:
        stop_event.set()
        if DISPLAY_WINDOWS:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
