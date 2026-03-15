import os
import json
import shutil
import subprocess


def create_abv_project(path, name="MyProject"):
    bundle = os.path.join(path, f"{name}.abv")

    os.makedirs(bundle, exist_ok=True)
    os.makedirs(os.path.join(bundle, "media"), exist_ok=True)

    project = {
        "name": name,
        "version": "1.0",
        "media": []
    }

    with open(os.path.join(bundle, "project.json"), "w") as f:
        json.dump(project, f, indent=2)

    # ----- Optional Finder icon for the bundle -----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_source = os.path.join(script_dir, "ABVDocument.icns")

    if os.path.exists(icon_source):
        icon_dest = os.path.join(bundle, ".VolumeIcon.icns")
        shutil.copy(icon_source, icon_dest)

        try:
            subprocess.run(["SetFile", "-a", "C", bundle], check=False)
        except Exception:
            pass

    print("ABView project created:", bundle)


if __name__ == "__main__":
    create_abv_project(".")