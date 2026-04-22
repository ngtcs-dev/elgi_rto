# plate_saver.py
import cv2
import os
import time
from datetime import datetime


class PlateSaver:
    def __init__(self, captures_dir: str, cooldown_seconds: int = 30):
        self.captures_dir    = captures_dir
        self.cooldown_seconds = cooldown_seconds
        self._last_saved: dict[str, float] = {}   # plate_number → epoch time

        os.makedirs(self.captures_dir, exist_ok=True)
        print(f"[SAVER] Images will be saved to: {os.path.abspath(self.captures_dir)}")

    def _is_on_cooldown(self, plate_number: str) -> bool:
        last = self._last_saved.get(plate_number, 0)
        return (time.time() - last) < self.cooldown_seconds

    def save(self, frame, plate_number: str, camera_name: str) -> str | None:
        """
        Save an annotated full frame when a plate is confirmed.
        Returns the saved file path, or None if on cooldown.
        """
        if self._is_on_cooldown(plate_number):
            print(f"[SAVER] Skipping {plate_number} — cooldown active.")
            return None

        now           = datetime.now()
        filename = f"{plate_number}.jpg"
        filepath      = os.path.join(self.captures_dir, filename)

        # annotated = self._add_overlay(frame.copy(), now, plate_number, camera_name)
        # cv2.imwrite(filepath, annotated)
        cv2.imwrite(filepath, frame)  # plain frame, no overlay

        self._last_saved[plate_number] = time.time()
        print(f"[SAVER] ✅ Saved: {filepath}")
        return filepath

    def _add_overlay(self, frame, dt: datetime, plate_number: str, camera_name: str):
        """
        Burn a semi-transparent dark banner at the bottom with:
          • Plate number (large)
          • Date & time
          • Camera name
        """
        h, w = frame.shape[:2]

        # ── Semi-transparent dark banner ──────────────────────────────────────
        banner_h = 90
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font       = cv2.FONT_HERSHEY_DUPLEX
        color_plate = (0, 230, 255)   # Amber/yellow — high visibility
        color_info  = (200, 200, 200) # Light grey for secondary info
        shadow      = (0, 0, 0)

        # ── Large plate number ────────────────────────────────────────────────
        plate_text  = f"PLATE: {plate_number}"
        plate_scale = 1.4
        plate_thick = 2
        plate_y     = h - banner_h + 40

        cv2.putText(frame, plate_text, (11, plate_y + 1),
                    font, plate_scale, shadow, plate_thick + 2)
        cv2.putText(frame, plate_text, (10, plate_y),
                    font, plate_scale, color_plate, plate_thick)

        # ── Secondary info (date | time | camera) ─────────────────────────────
        info_text  = (f"{dt.strftime('%d-%m-%Y')}   "
                      f"{dt.strftime('%H:%M:%S')}   "
                      f"CAM: {camera_name.upper()}")
        info_scale = 0.65
        info_thick = 1
        info_y     = h - 12

        cv2.putText(frame, info_text, (11, info_y + 1),
                    font, info_scale, shadow, info_thick + 1)
        cv2.putText(frame, info_text, (10, info_y),
                    font, info_scale, color_info, info_thick)

        return frame
