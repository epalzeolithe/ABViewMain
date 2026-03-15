from setuptools import setup

APP = ["BUNDLE.py"]

OPTIONS = {
    "argv_emulation": True,
    "iconfile": "data/ressources/ABV.icns",

    # Only include the Qt modules actually needed
    "includes": [
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
    ],

    # Explicitly exclude heavy Qt modules not used
    "excludes": [
        "PyQt5.QtQml",
        "PyQt5.QtQuick",
        "PyQt5.QtWebEngine",
        "PyQt5.QtWebEngineCore",
        "PyQt5.QtWebEngineWidgets",
        "PyQt5.QtMultimedia",
        "PyQt5.QtLocation",
        "PyQt5.QtPositioning",
        "PyQt5.QtTextToSpeech",
        "PyQt5.QtSql",
        "PyQt5.QtXmlPatterns",
    ],

    # Reduce bundle size
    "strip": True,
    "optimize": 2,

    "plist": {
        "CFBundleName": "ABview",
        "CFBundleIdentifier": "com.example.ABView",
        "CFBundleShortVersionString": "1.4",
        "CFBundleVersion": "1.4",
        "NSHighResolutionCapable": True,

        "UTExportedTypeDeclarations": [
            {
                "UTTypeIdentifier": "com.example.abv",
                "UTTypeDescription": "ABView Bundle",
                "UTTypeConformsTo": ["com.apple.package"],
                "UTTypeTagSpecification": {
                    "public.filename-extension": ["abv"]
                },
            }
        ],

        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "ABView Bundle",
                "CFBundleTypeRole": "Editor",
                "LSItemContentTypes": ["com.example.abv"],
                "LSTypeIsPackage": True,
            }
        ],
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)