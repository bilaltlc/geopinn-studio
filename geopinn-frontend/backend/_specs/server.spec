# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\telci\\Geopinn\\geopinn-backend\\server.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\telci\\Geopinn\\geopinn-backend\\engines', 'engines')],
    hiddenimports=['engines.gravity_prism', 'engines.magnetic_prism', 'engines.csamt_1d', 'engines.harmonica_validation', 'engines.fvm_core', 'engines.gravity_fvm', 'engines.magnetic_fvm', 'engines.petrophysics', 'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'scipy.ndimage', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.interpolate', 'fastapi', 'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'pydantic', 'numpy', 'harmonica'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorflow_core', 'keras', 'tensorboard', 'tf2onnx', 'jax', 'flax'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server',
)
