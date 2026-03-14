#!/usr/bin/env python3
import subprocess

# -------- CONFIG --------
X4_INSV_1 = "data/VID_20260221_091717_00_050.insv"
X4_INSV_2 = "data/VID_20260221_091717_00_051.insv"


def convert_insv(input_file, prefix):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-stats_period", "1",
        "-y",
        "-i", input_file,

        "-filter_complex",
        "[0:v:0][0:v:1]hstack,"
        "v360=input=dfisheye:output=hammer:ih_fov=193:iv_fov=193[v];"
        "[v]split=2[vf][vb];"
        "[vf]v360=input=hammer:output=hammer:yaw=0:w=1920:h=1080,"
        "crop=1200:675,scale=1920:1080:flags=lanczos[front];"
        "[vb]v360=input=hammer:output=hammer:yaw=180:w=1920:h=1080,"
        "crop=1080:608,scale=1920:1080:flags=lanczos[back]",

        "-map", "[front]",
        "-map", "0:a",
        # ter"-t", "5",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        "-b:a", "192k",
        f"{prefix}_front.mp4",

        "-map", "[back]",
        "-map", "0:a",
        "-t", "5",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        "-b:a", "192k",
        f"{prefix}_back.mp4",
    ]

    print(f"\nConversion : {input_file}")
    subprocess.run(cmd, check=True)


def main():
    convert_insv(X4_INSV_1, "x4_1")
    convert_insv(X4_INSV_2, "x4_2")

    print("\nConcat front videos...")
    with open("front_list.txt", "w") as f:
        f.write("file 'x4_1_front.mp4'\n")
        f.write("file 'x4_2_front.mp4'\n")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "front_list.txt",
        "-c", "copy",
        "front_merged.mp4"
    ], check=True)

    print("\nConcat back videos...")
    with open("back_list.txt", "w") as f:
        f.write("file 'x4_1_back.mp4'\n")
        f.write("file 'x4_2_back.mp4'\n")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "back_list.txt",
        "-c", "copy",
        "back_merged.mp4"
    ], check=True)


if __name__ == "__main__":
    main()