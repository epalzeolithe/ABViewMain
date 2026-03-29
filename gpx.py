# -------- CONFIG --------
CONSOLE_WINDOW = False

import os,re
from datetime import datetime, timedelta, timezone
import subprocess
from pymediainfo import MediaInfo
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
from PyQt5.QtWidgets import QApplication, QTextEdit
import gpxpy


# -------- SUBPROCESS STREAM (REAL-TIME) --------
def run_cmd_stream(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

def get_last_two_insv_files(directory):
    pattern = re.compile(r"^VID_.*?(\d{3})\.insv$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    # Trier par index croissant
    files_with_index.sort(key=lambda x: x[0])

    if not files_with_index:
        return None, "none.insv"

    if len(files_with_index) == 1:
        last_file = files_with_index[0][1]
        return last_file, "none.insv"

    # Avant-dernier et dernier
    second_last = files_with_index[-2][1]
    last = files_with_index[-1][1]

    return second_last, last


from ver import __version__

# -------- INPUT FILES

SUBDIR="data/raw/"
TMP=SUBDIR+"temp/"
X4_INSV_1, X4_INSV_2 = get_last_two_insv_files(SUBDIR)
EXIFTOOL = "data/ressources/exiftool"
EXIFFMT = "data/ressources/gpx.fmt"
MAINDIR="/Users/drax/Down/ABViewMain/"

def get_bundle_name_from_insv(path):
    name = os.path.basename(path)
    # attendu : VID_20260221_091717_00_050.insv
    parts = name.split("_")
    date = parts[1]  # 20260221
    time = parts[2]  # 091717
    return f"Vol_{date[:4]}_{date[4:6]}_{date[6:8]}.abv"

OUTPUT = "data/"+get_bundle_name_from_insv(X4_INSV_1)+"/merged_data.csv"

# ======================================================
# MP4 creation date (UTC)
# ======================================================
def get_mp4_creation_datetime(path):
    media_info = MediaInfo.parse(path)
    for track in media_info.tracks:
        if track.track_type == "General" and track.encoded_date:
            s = track.encoded_date.strip().replace("UTC", "").strip()
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
    raise RuntimeError(f"Date de création MP4 introuvable : {path}")

def export_EXIFTOOL_GPX_from_insv(insv_path):
    PATH = os.getcwd() + "/"
    input_file = PATH + SUBDIR + insv_path
    output_file = PATH + TMP + insv_path + ".gpx"
    cmd = [
        PATH + EXIFTOOL,
        "-ee",
        "-p", EXIFFMT,
        input_file]
    print("Running EXIFTOOL:", " ".join(cmd))
    with open(output_file, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    print("GPX créé:", output_file)

def read_EXIFTOOL_GPX(gpx_file):
    with open(gpx_file, "r") as f:
        gpx = gpxpy.parse(f)

    data = []

    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                data.append({
                    "timestamp": p.time.strftime("%Y-%m-%d %H:%M:%S") if p.time else None,
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "alt": round(p.elevation * 3.28084,0)
                })

    df = pd.DataFrame(data)

    # nettoyage optionnel
    df = df.dropna(subset=["timestamp", "lat", "lon", "alt"])
    return df

def get_datas_from_insv(insv):
    # insv > BB tool > export gpx
    export_EXIFTOOL_GPX_from_insv(insv)
    print("Export GPX done")
    gbb_df = read_EXIFTOOL_GPX(TMP+insv+".gpx")
    gbb_df.to_csv(TMP+insv+".gpx.csv", index=True, encoding="utf-8")
    return gbb_df

def main():
    Path(TMP).mkdir(parents=True, exist_ok=True)

    print("**********************************************************************")
    print("GPS extraction")
    print("X4_INSV_1:", X4_INSV_1)
    print("X4_INSV_2:", X4_INSV_2)
    if os.path.exists(SUBDIR+X4_INSV_1):
        print("Loading Datas from .insv")
        xdf = get_datas_from_insv(X4_INSV_1)
        print("Datas from .insv loaded")
    else:
        print(".insv file not found")

    if os.path.exists(SUBDIR+X4_INSV_2):
        print("Loading Datas from 2nd insv")
        xdf2 = get_datas_from_insv(X4_INSV_2)
        print("Datas from 2nd .insv loaded")
    else:
        print("2nd .insv file not found")

    print("Done.")

if __name__ == "__main__":

    if CONSOLE_WINDOW:
        app = QApplication(sys.argv)

        console = ConsoleWindow()
        console.show()

        stream = ConsoleStream()
        stream.new_text.connect(console.append_text)

        sys.stdout = stream
        sys.stderr = stream

        worker = Worker()
        worker.start()

        sys.exit(app.exec_())

    else:
        # mode console simple (pas de Qt)
        try:
            main()
        except Exception as e:
            print("Erreur :", e)
        sys.exit(0)
