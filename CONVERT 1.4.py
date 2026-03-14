#!/usr/bin/env python3
import subprocess

def build_ffmpeg_cmd(input1, input2, front_out, back_out, video_bitrate):
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y",
        "-t", "10", "-i", input1,
        "-t", "10", "-i", input2,
        "-filter_complex",
        """
[0:v:0][0:v:1]hstack[v0];
[1:v:0][1:v:1]hstack[v1];

[v0][v1]concat=n=2:v=1:a=0[v];

[0:a][1:a]concat=n=2:v=0:a=1[a];
[a]asplit=2[a1][a2];

[v]v360=input=dfisheye:output=hammer:ih_fov=193:iv_fov=193[vh];
[vh]split=2[vf][vb];

[vf]v360=input=hammer:output=hammer:yaw=0:w=1920:h=1080,
crop=1200:675,scale=1920:1080:flags=lanczos[front];

[vb]v360=input=hammer:output=hammer:yaw=180:w=1920:h=1080,
crop=1080:608,scale=1920:1080:flags=lanczos[back]
""",
        "-map", "[front]", "-map", "[a1]",
        "-c:v", "h264_videotoolbox", "-b:v", video_bitrate,
        "-c:a", "aac", "-b:a", "192k",
        front_out,

        "-map", "[back]", "-map", "[a2]",
        "-c:v", "h264_videotoolbox", "-b:v", video_bitrate,
        "-c:a", "aac", "-b:a", "192k",
        back_out,
    ]

# -------- CONFIG --------
X4_INSV_1 = "data/VID_20260221_091717_00_050.insv"
X4_INSV_2 = "data/VID_20260221_091717_00_051.insv"

def main():
    cmd = build_ffmpeg_cmd(X4_INSV_1, X4_INSV_2, "front_merged.mp4", "back_merged.mp4", "8M")

    print("Starting merging and conversion of : ",X4_INSV_1," and : ",X4_INSV_2)
    print(cmd)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e)

    print("Done.")

if __name__ == "__main__":
    main()