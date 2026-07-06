# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for STAR Analyzer.

Build with:
    pyinstaller star_analyzer.spec

Produces:  dist/STAR Analyzer/STAR Analyzer.exe  (--onedir mode)
"""

import os

block_cipher = None

# Check for icon file
icon_file = 'star_analyzer.ico'
if not os.path.exists(icon_file):
    icon_file = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'openpyxl',
        'pykalman',
        'scipy.spatial',
        'scipy.ndimage',
        'scipy.special',
        'scipy._lib',
        'scipy._lib._array_api',
        'scipy._lib.array_api_compat',
        'scipy._lib.array_api_compat._internal',
        'matplotlib',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.model_selection',
        'h5py',
        'cv2',
        'unittest',
        'unittest.case',
        'unittest.suite',
        'unittest.runner',
        'unittest.loader',
        'unittest.main',
        'unittest.signals',
        'unittest.result',
        'unittest.util',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'email',
        'xml',
        'xmlrpc',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='STAR Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='STAR Analyzer',
)
