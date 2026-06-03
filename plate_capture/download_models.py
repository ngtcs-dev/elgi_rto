# download_models.py
# Run this ONCE before building the exe.
# It forces PaddleOCR to download all required model files to your PC.
# After this script finishes, you can safely build the exe.

print("[INFO] Downloading PaddleOCR models... (this may take 2-5 minutes)")
print("[INFO] Models will be saved to your user folder under .paddleocr")
print()

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

print()
print("[INFO] ✅ All models downloaded successfully!")
print("[INFO] You can now run: pyinstaller main.spec")