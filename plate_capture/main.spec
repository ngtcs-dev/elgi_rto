# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata

# ======================
# Core OCR packages
# ======================
datas1, binaries1, hiddenimports1 = collect_all('paddleocr')
datas2, binaries2, hiddenimports2 = collect_all('paddle')

# Supporting dependencies
datas4, binaries4, hiddenimports4 = collect_all('pyclipper')
datas5, binaries5, hiddenimports5 = collect_all('shapely')
datas6, binaries6, hiddenimports6 = collect_all('skimage')
datas7, binaries7, hiddenimports7 = collect_all('imgaug')
datas8, binaries8, hiddenimports8 = collect_all('numpy')
datas9, binaries9, hiddenimports9 = collect_all('lmdb')

# 🔥 ADD THIS (fix for your error)
datas12, binaries12, hiddenimports12 = collect_all('setuptools')

# Optional but stabilizes builds
datas10, binaries10, hiddenimports10 = collect_all('PIL')
datas11, binaries11, hiddenimports11 = collect_all('yaml')

# ======================
# Metadata (important)
# ======================
metadata = (
    copy_metadata('imageio') +
    copy_metadata('imgaug') +
    copy_metadata('numpy') +
    copy_metadata('paddleocr') +
    copy_metadata('paddlepaddle') +
    copy_metadata('setuptools')   # 🔥 ADD THIS
)

# ======================
# Analysis
# ======================
a = Analysis(
    ['main.py'],
    pathex=[],

    binaries=(
        binaries1 + binaries2 +
        binaries4 + binaries5 + binaries6 +
        binaries7 + binaries8 + binaries9 +
        binaries10 + binaries11 +
        binaries12   # 🔥 ADD THIS
    ),

    datas=(
        datas1 + datas2 +
        datas4 + datas5 + datas6 +
        datas7 + datas8 + datas9 +
        datas10 + datas11 +
        datas12 +   # 🔥 ADD THIS
        metadata +
        [
            ('config.ini', '.'),
            ('imghdr.py', '.'),
        ]
    ),

    hiddenimports=(
        hiddenimports1 + hiddenimports2 +
        hiddenimports4 + hiddenimports5 +
        hiddenimports6 + hiddenimports7 +
        hiddenimports8 + hiddenimports9 +
        hiddenimports10 + hiddenimports11 +
        hiddenimports12 +   # 🔥 ADD THIS
        [
            # Core
            'cv2',
            'paddleocr',
            'paddle',

            # OCR deps
            'pyclipper',
            'shapely',
            'skimage',
            'imgaug',

            # NumPy internals
            'numpy',
            'numpy.core',
            'numpy.core.multiarray',

            # Storage
            'lmdb',

            # Python 3.12 fix
            'imghdr',

            # Misc
            'PIL',
            'yaml',

            # 🔥 CRITICAL
            'setuptools',
        ]
    ),

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # ✅ Keep Cython excluded
    excludes=[
        'Cython',
        'Cython.Compiler',
        'Cython.Compiler.Code',
    ],

    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ----------------------
# EXE
# ----------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

# ----------------------
# Final bundle
# ----------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='main',
)