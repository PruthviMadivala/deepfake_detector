import json
import subprocess

# Set your ffprobe path
FFPROBE_PATH = r"C:\Users\suman\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffprobe.exe"


def get_audio_timestamps(video_path):
    """
    Extract audio packet PTS timestamps using ffprobe (JSON output).
    Works reliably on all Windows builds.
    """
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "quiet",
            "-print_format", "json",
            "-show_packets",
            "-select_streams", "a",
            video_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Parse JSON
        data = json.loads(result.stdout)

        timestamps = []
        for pkt in data.get("packets", []):
            if "pts_time" in pkt:
                timestamps.append(float(pkt["pts_time"]))

        return timestamps

    except Exception as e:
        print("⚠ Error reading audio:", e)
        return []


def audio_video_sync_score(timestamps):
    """
    Calculate audio-video sync score.
    Lower avg gap → better synchronization.
    """
    if len(timestamps) < 5:
        return -1  # No audio or failed extraction

    diffs = [abs(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]

    avg_gap = sum(diffs) / len(diffs)

    if avg_gap < 0.05:
        return 1    # Perfect sync
    elif avg_gap < 0.15:
        return 0.5  # Minor issues
    else:
        return 0    # Bad sync → deepfake chance


if __name__ == "__main__":
    video = r"C:\Users\suman\Desktop\deepfake-detector\test\trump.mp4"

    print(f"Checking audio sync for: {video}\n")

    timestamps = get_audio_timestamps(video)
    score = audio_video_sync_score(timestamps)

    print("Audio–Video timestamps extracted:", len(timestamps))
    print("Audio–Video sync score:", score)
