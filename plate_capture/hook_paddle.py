# hook_paddle.py  — PyInstaller runtime hook
# This runs BEFORE main.py starts.
# Fixes paddleocr path resolution when running as a frozen .exe

import os
import sys

if getattr(sys, 'frozen', False):
    # _internal folder — where all bundled files live
    internal = sys._MEIPASS

    # paddleocr.py uses:
    #   __dir__ = os.path.dirname(__file__)
    #   tools = _import_file('tools', os.path.join(__dir__, 'tools/__init__.py'))
    #
    # When frozen, __file__ for paddleocr.py resolves to _internal\paddleocr.py
    # so __dir__ = _internal  and it looks for  _internal\tools\__init__.py
    # That means tools/, ppocr/, ppstructure/ MUST be directly in _internal/
    # which is what our spec does (dest = "tools", "ppocr", "ppstructure")

    # Add _internal to sys.path so that:
    #   import ppocr        works
    #   import ppstructure  works
    #   import tools        works
    if internal not in sys.path:
        sys.path.insert(0, internal)

    # Point PaddleOCR model downloads to bundled models folder
    paddle_ocr_home = os.path.join(internal, '.paddleocr')
    os.environ['PADDLE_OCR_HOME']  = paddle_ocr_home
    os.environ['HOME']             = internal

    # Suppress noisy paddle logs
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['FLAGS_use_paddle_jit'] = '0'