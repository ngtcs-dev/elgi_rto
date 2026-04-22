# main.py  — PaddleOCR 2.7.3  (state-code validated, vote-to-confirm)
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

# ── INIT ──────────────────────────────────────────────────────────────────────
ocr_engine        = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
plate_saver       = PlateSaver(captures_dir=CAPTURES_DIR, cooldown_seconds=COOLDOWN_SECONDS)
connected_clients = []
detection_lock    = threading.Lock()

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
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    per_pass, all_boxes = [], []
    for img in [frame, enhanced]:
        text, boxes = run_ocr(img)
        if text:
            per_pass.append(text)
            all_boxes.extend(boxes)
    return per_pass, all_boxes

# ── PLATE MATCHING ────────────────────────────────────────────────────────────
_STD_LOOSE  = re.compile(r'[A-Z0-9]{2}\d{0,2}[A-Z0-9]{1,3}\d{3,4}')
_BH_LOOSE   = re.compile(r'\d{2}BH\d{3,4}[A-Z]{1,2}')
_STD_STRICT = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')
_BH_STRICT  = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')

_D2L = {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G'}
_L2D = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0','D':'0'}

def _fl(ch): return _D2L.get(ch, ch)   # force-letter
def _fd(ch): return _L2D.get(ch, ch)   # force-digit

def normalize_plate(raw: str) -> str | None:
    """
    Normalize OCR noise using position rules.
    Also validates state code — returns None if state is not a known Indian code.
    """
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
        # State (pos 0,1) → letters
        t[0] = _fl(t[0]); t[1] = _fl(t[1])
        # District (pos 2,3) → digits
        t[2] = _fd(t[2]); t[3] = _fd(t[3])
        # Series letters (pos 4 to n-4) → letters
        for i in range(4, max(4, n - 4)): t[i] = _fl(t[i])
        # Number (last 4) → digits
        for i in range(max(0, n - 4), n): t[i] = _fd(t[i])

        # ── STATE CODE VALIDATION ─────────────────────────────────────────
        state = ''.join(t[:2])
        snapped = snap_state_code(state)
        if snapped is None:
            return None          # Not a real Indian state — discard plate
        t[0] = snapped[0]
        t[1] = snapped[1]

    return ''.join(t)

def pad_to_4_digits(plate: str) -> str:
    """If last group is 3 digits (OCR dropped one), append 0."""
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

    # Length constraint
    if not (6 <= len(text) <= 10):
        return False

    # ── 1. BH SERIES CHECK (FIRST PRIORITY) ─────────────────────
    # Format: 22BH1234AA
    if len(text) >= 8:
        if text[:2].isdigit() and text[2:4] == "BH":
            i = 4

            # Next must be digits (3–4)
            digit_count = 0
            while i < len(text) and text[i].isdigit() and digit_count < 4:
                digit_count += 1
                i += 1

            if digit_count < 3:
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

    # Optional letters (0–2)
    letter_count = 0
    while i < len(text) and text[i].isalpha() and letter_count < 2:
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
                norm = pad_to_4_digits(norm)
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm
            for m in _BH_LOOSE.findall(cand):
                norm = normalize_plate(m)
                if norm is None:
                    continue
                bm = re.match(r'^(\d{2}BH)(\d{3})([A-Z]{1,2})$', norm)
                if bm:
                    norm = bm.group(1) + bm.group(2) + '0' + bm.group(3)
                if is_valid_plate_strict(norm):
                    results[norm[-6:]] = norm

    for text in pass_texts:
        _try(text)
    _try(" ".join(pass_texts))

    return list(results.values())

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
    # cores already saved this session (checked against plate_saver cooldown too)
    saved_cores: set[str]          = set()

    while True:
        try:
            item = fq.get(timeout=5)
        except queue.Empty:
            continue
        if item is None:
            break

        frame, cap_time = item

        per_pass, all_boxes = ocr_full_frame(frame)
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
                if core in saved_cores:
                    print(f"[{camera_name}] ⏳ Already saved core {core} this session")
                    core_hits[core] = 0
                    continue

                saved = plate_saver.save(frame=frame, plate_number=best,
                                         camera_name=camera_name)
                if saved:
                    print(f"[{camera_name}] ✅ SAVED: {best}  →  {saved}")
                    saved_cores.add(core)
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

        # Remove cores not seen recently (older than 15 frames worth of ocr)
        stale = [c for c, h in core_hits.items() if h == 0 and c not in saved_cores]
        for c in stale:
            core_votes.pop(c, None)
            core_hits.pop(c, None)

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
        'last_raw_text': "",
        'last_boxes':    [],
        'last_plates':   [],
        'best_plate':    None,
        'hits':          {},
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
                hits          = shared.get('hits', {}),
                required_hits = REQUIRED_HITS,
                raw_text      = shared.get('last_raw_text', ""),
                best_plate    = shared.get('best_plate'),
                next_ocr_in   = max(0.0, OCR_INTERVAL_SECONDS - (time.time() - last_ocr_time)),
                fps           = fps,
                cam           = name,
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