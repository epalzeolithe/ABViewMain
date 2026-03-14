from setuptools import setup

APP = ["ABView 1.3.py"]

OPTIONS = {
    "argv_emulation": True,

    # modules dynamiques que py2app ne détecte pas toujours
    "includes": [
        "objc",
        "Foundation",
        "CoreFoundation",
        "CoreMedia",
        "AVFoundation",
        "ScreenCaptureKit",

        "rubicon",
        "rubicon.objc",
        "rubicon.objc.api",
        "rubicon.objc.runtime",

        "wgpu.backends.auto",
        "wgpu.backends.wgpu_native",

        "rendercanvas.qt",
    ],

    # packages lourds mais nécessaires
    "packages": [
        "numpy",
        "pandas",
        "pygfx",
        "wgpu",
        "rubicon",
    ],

    # modules inutiles dans une app macOS
    "excludes": [
        "tkinter",
        "pytest",
        "test",
        "unittest",
    ],

    "plist": {
        "CFBundleName": "ABView",
        "CFBundleIdentifier": "com.drax.abview",
        "CFBundleShortVersionString": "1.3",
        "CFBundleVersion": "1.3",
        "NSHighResolutionCapable": True,
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)