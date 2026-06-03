from cryptography.fernet import Fernet
"""
license_gate.py  —  Hardware-locked license activation for ANPR system
Flow:
  Step 1 — Client enters device name, client name, and location →
            app combines MAC + all three fields,
            encrypts it and shows as Device ID string.
  Step 2 — Client copies Device ID → sends to admin portal.
  Step 3 — Admin portal decrypts it, generates a signed token.
  Step 4 — Client pastes token → validated (MAC + device name + expiry).
"""

import sys, uuid, json, base64, hashlib, hmac, re, argparse
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE       = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
LICENSE_DAT = _BASE / "license.dat"
KEY_FILE    = _BASE / ".lk"

# ── Shared secret — must match TOKEN_SECRET in portal crypto.js ──────────────
_TOKEN_SECRET = "ANPR_L1c3ns3_S3cr3t_K3y_2025_@Secure"


# ══════════════════════════════════════════════════════════════════════════════
# Device helpers
# ══════════════════════════════════════════════════════════════════════════════

def _xor_with_secret(data: bytes) -> bytes:
    """XOR each byte with repeating secret bytes — same logic as JS."""
    secret = _TOKEN_SECRET.encode()
    return bytes(b ^ secret[i % len(secret)] for i, b in enumerate(data))


def get_mac() -> str:
    """Return MAC as AA:BB:CC:DD:EE:FF"""
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    return ":".join(mac[i:i+2] for i in range(0, 12, 2))


def build_device_id(mac: str, name: str, client_name: str, location: str) -> str:
    """
    Combine MAC + device name + client name + location into JSON,
    XOR with secret, return base64url string.
    Matches decryptDeviceId() in crypto.js exactly.
    """
    payload = json.dumps({
        "mac":         mac,
        "name":        name.strip(),
        "client_name": client_name.strip(),
        "location":    location.strip(),
    }).encode()
    xored = _xor_with_secret(payload)
    return base64.urlsafe_b64encode(xored).decode().rstrip("=")


def decrypt_device_id(device_id_str: str) -> dict:
    """
    Reverse of build_device_id.
    Returns {"mac": "...", "name": "...", "client_name": "...", "location": "..."}
    """
    try:
        padded = device_id_str + "=" * (-len(device_id_str) % 4)
        raw    = base64.urlsafe_b64decode(padded)
        plain  = _xor_with_secret(raw)
        return json.loads(plain)
    except Exception:
        raise ValueError("Invalid or corrupted Device ID.")


# ══════════════════════════════════════════════════════════════════════════════
# Token validation
# ══════════════════════════════════════════════════════════════════════════════

def _sha256hex(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()


def _b64url_decode(s: str) -> str:
    s = s.replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    return base64.b64decode(s).decode("utf-8")


def decode_token(token: str):
    parts = token.strip().split(".")
    if len(parts) != 2:
        raise ValueError("Invalid key format.")
    b64_payload, sig = parts
    try:
        payload_str = _b64url_decode(b64_payload)
        payload     = json.loads(payload_str)
    except Exception:
        raise ValueError("Key is corrupted or tampered.")
    expected_sig = _sha256hex(_TOKEN_SECRET + payload_str)[:32]
    if not hmac.compare_digest(expected_sig, sig):
        raise ValueError("Invalid key. Signature does not match.")
    return payload, payload_str


def normalize_mac(raw: str) -> str:
    clean = re.sub(r"[^A-F0-9]", "", raw.upper())
    return ":".join(clean[i:i+2] for i in range(0, 12, 2))


def validate_token(token: str, expected_mac: str, expected_name: str) -> tuple:
    """
    Returns (success: bool, message: str)
    Checks: format → signature → device_id → device_name → expiry
    """
    try:
        payload, _ = decode_token(token)
    except ValueError as e:
        return False, str(e)

    for field in ("device_id", "device_name", "user", "issued_at", "expires"):
        if field not in payload:
            return False, "Invalid key. Missing required fields."

    # ── MAC check ─────────────────────────────────────────────────────────────
    token_mac = normalize_mac(payload["device_id"])
    if expected_mac.upper() != token_mac:
        return False, (
            f"Wrong device.\n"
            f"Key issued for: {token_mac}\n"
            f"This machine:   {expected_mac}"
        )

    # ── Name check ────────────────────────────────────────────────────────────
    if payload["device_name"].strip().lower() != expected_name.strip().lower():
        return False, (
            f"Device name mismatch.\n"
            f"Key issued for: '{payload['device_name']}'\n"
            f"This machine:   '{expected_name}'"
        )

    # ── Expiry check ──────────────────────────────────────────────────────────
    try:
        expires = datetime.fromisoformat(payload["expires"].replace("Z", "+00:00"))
        now     = datetime.now(timezone.utc)
        if now > expires:
            return False, (
                "Key has expired. Please generate a new key\n"
                f"from the admin portal and try again.\n"
                f"(Expired at {expires.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
            )
    except Exception:
        return False, "Invalid expiry date in key."

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# License persistence
# ══════════════════════════════════════════════════════════════════════════════

def _get_fernet() -> Fernet:
    if KEY_FILE.exists():
        key = KEY_FILE.read_bytes()
    else:
        key = Fernet.generate_key()
        KEY_FILE.write_bytes(key)
    return Fernet(key)


def save_license(token: str, payload: dict, device_name: str,
                 client_name: str = "", location: str = ""):
    f    = _get_fernet()
    data = json.dumps({
        "mac":         payload["device_id"],
        "device_name": device_name,
        "client_name": client_name,
        "location":    location,
        "token":       token,
        "activated":   True,
    }).encode()
    LICENSE_DAT.write_bytes(f.encrypt(data))


def load_license() -> dict | None:
    if not LICENSE_DAT.exists() or not KEY_FILE.exists():
        return None
    try:
        f = _get_fernet()
        return json.loads(f.decrypt(LICENSE_DAT.read_bytes()))
    except Exception:
        return None


def is_activated() -> bool:
    lic = load_license()
    if not lic or not lic.get("activated"):
        return False
    return normalize_mac(lic.get("mac", "")) == get_mac()


# ══════════════════════════════════════════════════════════════════════════════
# GUI — Step 1: Enter device name, client name, and location
# ══════════════════════════════════════════════════════════════════════════════

class NameDialog:
    def __init__(self):
        self.name        = None   # device name
        self.client_name = None
        self.location    = None

        self.root = tk.Tk()
        self.root.title("License Activation")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_ui()

        # Let tkinter calculate the required size, then centre on screen
        self.root.update_idletasks()
        w = max(self.root.winfo_reqwidth(),  440)
        h = max(self.root.winfo_reqheight(), 380)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.root.mainloop()

    def _build_ui(self):
        tk.Label(self.root, text="License Activation",
                 font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))

        tk.Label(self.root,
                 text="Enter the details below.\n"
                      "These will be tied to your license.",
                 font=("Segoe UI", 9), justify="center", fg="#444").pack(pady=(0, 10))

        # ── Device Name ───────────────────────────────────────────────────────
        tk.Label(self.root, text="Device Name",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=20)
        self._name_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self._name_var,
                 font=("Segoe UI", 10), relief="solid", bd=1).pack(
                     fill="x", padx=20, pady=(4, 8), ipady=5)

        # ── Client Name ───────────────────────────────────────────────────────
        tk.Label(self.root, text="Client Name",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=20)
        self._client_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self._client_var,
                 font=("Segoe UI", 10), relief="solid", bd=1).pack(
                     fill="x", padx=20, pady=(4, 8), ipady=5)

        # ── Location ──────────────────────────────────────────────────────────
        tk.Label(self.root, text="Location",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=20)
        self._location_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self._location_var,
                 font=("Segoe UI", 10), relief="solid", bd=1).pack(
                     fill="x", padx=20, pady=(4, 4), ipady=5)

        self._err_lbl = tk.Label(self.root, text="", font=("Segoe UI", 8), fg="red")
        self._err_lbl.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(10, 18))   # bottom padding keeps buttons visible

        tk.Button(btn_frame, text="Cancel", width=10,
                  command=self._on_cancel).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Next", width=10,
                  command=self._on_next).pack(side="left", padx=8)

        self.root.bind("<Return>", lambda e: self._on_next())

    def _on_next(self):
        name        = self._name_var.get().strip()
        client_name = self._client_var.get().strip()
        location    = self._location_var.get().strip()

        if not name:
            self._err_lbl.config(text="Please enter a device name.")
            return
        if not client_name:
            self._err_lbl.config(text="Please enter a client name.")
            return
        if not location:
            self._err_lbl.config(text="Please enter a location.")
            return

        self.name        = name
        self.client_name = client_name
        self.location    = location
        self.root.destroy()

    def _on_cancel(self):
        self.name        = None
        self.client_name = None
        self.location    = None
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# GUI — Step 2: Show Device ID + paste Access Key
# ══════════════════════════════════════════════════════════════════════════════

class ActivationDialog:
    def __init__(self, device_name: str, client_name: str, location: str):
        self.activated   = False
        self.device_name = device_name
        self.client_name = client_name
        self.location    = location
        self.mac         = get_mac()
        self.device_id   = build_device_id(self.mac, device_name, client_name, location)

        self.root = tk.Tk()
        self.root.title("License Activation")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        w, h = 520, 340
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self._build_ui()
        self.root.mainloop()

    def _build_ui(self):
        tk.Label(self.root, text="License Activation",
                 font=("Segoe UI", 13, "bold")).pack(pady=(18, 4))

        tk.Label(self.root,
                 text="Share your Device ID with the administrator to receive an Access Key.",
                 font=("Segoe UI", 9), justify="center", fg="#444").pack(pady=(0, 8))

        # ── Info summary ──────────────────────────────────────────────────────
        info_frame = tk.Frame(self.root, bg="#f0f4f8", bd=1, relief="solid")
        info_frame.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(info_frame,
                 text=f"Device: {self.device_name}   |   Client: {self.client_name}   |   Location: {self.location}",
                 font=("Segoe UI", 8), fg="#555", bg="#f0f4f8").pack(pady=5, padx=8)

        # ── Device ID (encrypted) ─────────────────────────────────────────────
        tk.Label(self.root, text="Device ID",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=16)

        dev_frame = tk.Frame(self.root)
        dev_frame.pack(fill="x", padx=16, pady=(2, 4))

        dev_var = tk.StringVar(value=self.device_id)
        tk.Entry(dev_frame, textvariable=dev_var, state="readonly",
                 font=("Courier New", 7), relief="solid", bd=1).pack(
                     side="left", fill="x", expand=True, ipady=4)

        tk.Button(dev_frame, text="Copy", width=8,
                  command=self._copy_device_id).pack(side="left", padx=(6, 0))

        self._copy_lbl = tk.Label(self.root, text="", font=("Segoe UI", 8), fg="green")
        self._copy_lbl.pack()

        # ── Access Key ────────────────────────────────────────────────────────
        tk.Label(self.root, text="Access Key",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=16)

        self._key_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self._key_var,
                 font=("Segoe UI", 9), relief="solid", bd=1).pack(
                     fill="x", padx=16, pady=(2, 4), ipady=4)

        self._status_lbl = tk.Label(self.root, text="", font=("Segoe UI", 8),
                                    fg="red", wraplength=480, justify="center")
        self._status_lbl.pack(pady=(2, 0))

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(8, 0))

        tk.Button(btn_frame, text="Cancel", width=10,
                  command=self._on_cancel).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Submit", width=10,
                  command=self._on_submit).pack(side="left", padx=8)

    def _copy_device_id(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.device_id)
        self._copy_lbl.config(text="Copied to clipboard.")
        self.root.after(2000, lambda: self._copy_lbl.config(text=""))

    def _on_submit(self):
        key = self._key_var.get().strip()
        self._status_lbl.config(text="")

        if not key:
            self._status_lbl.config(text="Please enter the access key.")
            return

        ok, message = validate_token(key, self.mac, self.device_name)

        if ok:
            payload, _ = decode_token(key)
            save_license(key, payload, self.device_name,
                         self.client_name, self.location)
            messagebox.showinfo("Activated", "License activated successfully!")
            self.activated = True
            self.root.destroy()
        else:
            self._status_lbl.config(text=message)

    def _on_cancel(self):
        self.activated = False
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def check_or_activate() -> bool:
    if is_activated():
        return True

    # Step 1 — get device name, client name, location
    name_dlg = NameDialog()
    if not name_dlg.name:
        return False

    # Step 2 — show device ID + key entry
    act_dlg = ActivationDialog(
        device_name=name_dlg.name,
        client_name=name_dlg.client_name,
        location=name_dlg.location,
    )
    return act_dlg.activated


if __name__ == "__main__":
    ok = check_or_activate()
    print("Activated:", ok)