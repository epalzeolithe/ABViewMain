from setuptools import setup

APP = ['ABView 1.3.py']

OPTIONS = {
    "argv_emulation": True,


    "includes": [
        "objc",
        "rubicon.objc",
        "rubicon.objc.runtime",
        "rubicon.objc.api",
    ],


    "frameworks": [],   # important
    "excludes": ["rubicon", "py2app.bootstrap.rubicon", "objc", "py2app.bootstrap.objc"],
    "plist": {
        "CFBundleName": "ABView",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "ABView Project",
                "CFBundleTypeExtensions": ["abview"],
                "CFBundleTypeRole": "Editor",
                "LSTypeIsPackage": True,
            }
        ],
    }
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)