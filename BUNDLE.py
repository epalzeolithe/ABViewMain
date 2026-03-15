import os
import plistlib
import subprocess


def create_abv_bundle(path, name="MyABView"):
    bundle_path = os.path.join(path, f"{name}.abv")
    contents = os.path.join(bundle_path, "Contents")
    macos = os.path.join(contents, "MacOS")
    resources = os.path.join(contents, "Resources")

    os.makedirs(macos, exist_ok=True)
    os.makedirs(resources, exist_ok=True)

    info = {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": "com.example.abview",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": name,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "11.0",
    }

    info_plist_path = os.path.join(contents, "Info.plist")
    with open(info_plist_path, "wb") as f:
        plistlib.dump(info, f)

    # executable placeholder
    exe_path = os.path.join(macos, name)
    with open(exe_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'ABView bundle launched'\n")

    os.chmod(exe_path, 0o755)

    # Force Finder to treat the extension as a package (bundle)
    try:
        subprocess.run([
            "xattr",
            "-w",
            "com.apple.FinderInfo",
            "00000000000000000010000000000000",
            bundle_path
        ], check=False)
    except Exception:
        pass

    print("Bundle created:", bundle_path)


if __name__ == "__main__":
    create_abv_bundle(".")