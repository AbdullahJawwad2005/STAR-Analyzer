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
        'openpyxl.styles.alignment',
        'openpyxl.styles.borders',
        'openpyxl.styles.fills',
        'openpyxl.styles.fonts',
        'openpyxl.styles.numbers',
        'pykalman',
        'scipy.spatial',
        'scipy.ndimage',
        'scipy.special',
        'scipy._lib',
        'scipy._lib._array_api',
        'scipy._lib.array_api_compat',
        'scipy._lib.array_api_compat._internal',
        'scipy.interpolate',
        'scipy.signal',
        'matplotlib',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.model_selection',
        'sklearn.metrics',
        'sklearn.inspection',
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        'h5py',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'PIL',
        'Pillow',
        'IPython',
        'ipykernel',
        'notebook',
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'nbformat',
        'nbconvert',
        'traitlets',
        'setuptools',
        'pkg_resources',
        'distutils',
        'unittest',
        'doctest',
        'pydoc',
        'test',
        'xmlrpc',
        'curses',
        'lib2to3',
        'email',
        'html',
        'http',
        'urllib',
        'ftplib',
        'imaplib',
        'smtplib',
        'poplib',
        'telnetlib',
        'socket',
    ],
    noarchive=False,
    optimize=1,
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
