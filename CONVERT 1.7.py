#!/usr/bin/env python3
from pymediainfo import MediaInfo
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import subprocess,os,re
import psutil
import time
import sys
from pathlib import Path

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

def get_last_GPS_log_file(directory):
    pattern = re.compile(r"^LOG.*?(\d{5})\.txt$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    if not files_with_index:
        return "none.txt"

    # Tri par index croissant
    files_with_index.sort(key=lambda x: x[0])

    # Retourne le dernier fichier
    return files_with_index[-1][1]


# -------- CONFIG --------
#X4_INSV_1 = "data/raw/VID_20260320_131559_00_053.insv"
#X4_INSV_2 = "data/raw/VID_20260320_131559_00_054.insv"
SUBDIR="data/raw/"
X4_INSV_1, X4_INSV_2 = get_last_two_insv_files(SUBDIR)
X4_INSV_1=SUBDIR+X4_INSV_1
X4_INSV_2=SUBDIR+X4_INSV_2


CROP_FRONT = "1200:675" #default 1200:675
CROP_BACK = "960:540" #default 1080:608

#Très large 1280:720
#Large 1200:675
#Moyen 1080:608
#Moyen+ 960:540
#Zoom 840:472
#Zoom fort 720:405
#Très zoom 640:360

SKIP_CONVERSION = False

def build_ffmpeg_cmd(input1, input2, front_out, back_out, video_bitrate):
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y",
        "-hwaccel", "videotoolbox",
        #"-t", "10",
        "-i", input1,
        #"-t", "10",
        "-i", input2,
        "-filter_complex",
        f"""
[0:v:0][0:v:1]hstack[v0];
[1:v:0][1:v:1]hstack[v1];

[v0][v1]concat=n=2:v=1:a=0[v];

[0:a][1:a]concat=n=2:v=0:a=1[a];
[a]asplit=2[a1][a2];

[v]v360=input=dfisheye:output=hammer:ih_fov=193:iv_fov=193[vh];
[vh]split=2[vf][vb];

[vf]v360=input=hammer:output=hammer:yaw=0:pitch=-25:w=1920:h=1080,
crop={CROP_FRONT},scale=1920:1080:flags=lanczos[front];

[vb]v360=input=hammer:output=hammer:yaw=180:w=1920:h=1080,
crop={CROP_BACK},scale=1920:1080:flags=lanczos[back]
""",
        "-map", "[front]", "-map", "[a1]",
        "-c:v", "h264_videotoolbox",
        "-b:v", video_bitrate,
        "-maxrate", video_bitrate,
        "-bufsize", "24M",
        "-profile:v", "high",
        "-g", "60",
        "-allow_sw", "1",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        front_out,

        "-map", "[back]", "-map", "[a2]",
        "-c:v", "h264_videotoolbox",
        "-b:v", video_bitrate,
        "-maxrate", video_bitrate,
        "-bufsize", "24M",
        "-profile:v", "high",
        "-g", "60",
        "-allow_sw", "1",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        back_out,]

def get_mp4_creation_datetime(path):
    media_info = MediaInfo.parse(path)
    for track in media_info.tracks:
        if track.track_type == "General" and track.encoded_date:
            s = track.encoded_date.strip().replace("UTC", "").strip()
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
    raise RuntimeError(f"Date de création MP4 introuvable : {path}")


def set_mp4_creation_datetime(path, dt):
    # si pas de fuseau → on suppose Europe/Paris
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Paris"))
    # FFmpeg attend généralement UTC → on encode explicitement en UTC avec Z
    ts = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = path + ".tmp.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", path,
        "-map_metadata", "0",
        "-metadata", f"creation_time={ts}",
        "-codec", "copy",
        tmp]
    subprocess.run(cmd, check=True)
    # remplace le fichier original
    import os
    os.replace(tmp, path)

def get_bundle_name_from_insv(path):
    name = os.path.basename(path)
    # attendu : VID_20260221_091717_00_050.insv
    parts = name.split("_")
    date = parts[1]  # 20260221
    time = parts[2]  # 091717
    return f"Vol_{date[:4]}_{date[4:6]}_{date[6:8]}.abv"

def main():
    bdl=get_bundle_name_from_insv(X4_INSV_1)
    pdl="data/"+bdl
    #crée le bundle
    Path(pdl).mkdir(parents=True, exist_ok=True)
    pdl=pdl+"/"
    print("Bundle : "+pdl)

    front=pdl+"front.mp4"
    back=pdl+"back.mp4"
    cmd = build_ffmpeg_cmd(X4_INSV_1, X4_INSV_2, back, front, "8M")
    print("Starting merging and conversion of : ",X4_INSV_1," and : ",X4_INSV_2)
    #print(cmd)

    if not SKIP_CONVERSION:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # initialise la mesure CPU
            psutil.cpu_percent(None)

            start = time.time()

            while process.poll() is None:
                cpu = psutil.cpu_percent(None)
                ram = psutil.virtual_memory().used / (1024**3)
                elapsed = int(time.time() - start)

                msg = f"Conversion... CPU {cpu:5.1f}% | RAM {ram:4.1f} GB | t={elapsed:4d}s"
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()

                time.sleep(1)

            process.wait()
        except subprocess.CalledProcessError as e:
            print("FFmpeg failed:", e)

    print()  # retour à la ligne après la barre dynamique

    #transfer date de création
    d= get_mp4_creation_datetime(X4_INSV_1)
    set_mp4_creation_datetime(front,d)
    set_mp4_creation_datetime(back,d)

    print("Done.")

if __name__ == "__main__":
    main()