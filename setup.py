from setuptools import setup

APP = ["ABView 1.4.py"]

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
    ],

    # modules inutiles dans une app macOS
    "excludes": [
        "tkinter",
        "pytest",
        "test",
        "unittest",
        "rubicon"   # ← IMPORTAN
    ],

    "iconfile": "ABVDocument.icns",
    "strip": True,
    "optimize": 2,

    "plist": {
    "CFBundleName": "ABView",
    "CFBundleIdentifier": "com.drax.abv",
    "CFBundleShortVersionString": "1.4",
    "CFBundleVersion": "1.4",
    "NSHighResolutionCapable": True,

    "CFBundleDocumentTypes": [
        {
            "CFBundleTypeName": "ABView Project",
            "CFBundleTypeRole": "Editor",
            "LSItemContentTypes": ["com.drax.abv.project"],
        }
    ],

    "UTExportedTypeDeclarations": [
        {
            "UTTypeIdentifier": "com.drax.abv.project",
            "UTTypeDescription": "ABView Project",
            "UTTypeConformsTo": ["public.data", "com.apple.package"],
            "UTTypeTagSpecification": {
                "public.filename-extension": ["abv"]
            }
        }
    ]
},
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)