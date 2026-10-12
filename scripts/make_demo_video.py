"""Assemble the vertical (1080x1920) LinkedIn demo video from the processed clips.

Run scripts/run_video.py on busy_street_night.mp4 and france_crosswalk_bollards.mp4 first, then:
    python scripts/make_demo_video.py
Output: data/outputs/linkedin_demo.mp4
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OUT_DIR = Path("data/outputs")
W, H, FPS = 1080, 1920, 30
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"  # ':' must be escaped inside ffmpeg filters
FONT_REGULAR = "C\\:/Windows/Fonts/segoeui.ttf"
REPO_URL = "github.com/Eya-Jmaa/blind-navigation-assistant"


def text(line: str, y: int, size: int, bold: bool = True, color: str = "white") -> str:
    """ffmpeg drawtext filter: one horizontally centered line of text."""
    line = line.replace(":", "\\:").replace("'", "’")
    return (f"drawtext=fontfile='{FONT if bold else FONT_REGULAR}':text='{line}':"
            f"fontsize={size}:fontcolor={color}:x=(w-text_w)/2:y={y}")


def card(lines: list[tuple[str, int, int, bool, str]], seconds: float, out: Path) -> None:
    """A dark title card with silent audio (every segment needs an audio track to concatenate)."""
    filters = ",".join(text(*line) for line in lines)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=0x101418:s={W}x{H}:r={FPS}:d={seconds}",
                    "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={seconds}",
                    "-vf", filters, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(out)], check=True)


def clip(src: Path, video_filter: str, out: Path) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                    "-filter_complex", video_filter, "-map", "[v]", "-map", "0:a",
                    "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-c:a", "aac", "-ar", "44100", "-ac", "2", str(out)], check=True)


def main():
    tmp = Path(tempfile.mkdtemp())
    parts = [tmp / f"{i}.mp4" for i in range(4)]

    card([("I built an AI assistant", 700, 72, True, "white"),
          ("that helps blind people", 800, 72, True, "white"),
          ("walk safely", 900, 72, True, "0x4FC3F7"),
          ("Sound on", 1150, 44, False, "0xB0BEC5")], 3, parts[0])

    # Busy street (side-by-side 2560x720): camera on top, depth below, titles around.
    clip(OUT_DIR / "busy_street_night_demo.mp4",
         f"[0:v]split[a][b];[a]crop=iw/2:ih:0:0,scale={W}:-2[cam];"
         f"[b]crop=iw/2:ih:iw/2:0,scale={W}:-2[dep];"
         f"[cam][dep]vstack,pad={W}:{H}:0:(oh-ih)/2:color=0x101418,"
         + text("What the AI sees", 150, 60) + ","
         + text("Camera: objects, distance, direction, time-to-collision", 250, 36, False, "0xB0BEC5") + ","
         + text("Depth: distance of every pixel (bright = close)", 1640, 36, False, "0xB0BEC5")
         + "[v]", parts[1])

    # Crosswalk (portrait, camera + depth stacked): keep the camera half, fill the frame.
    clip(OUT_DIR / "france_crosswalk_bollards_demo.mp4",
         f"[0:v]crop=iw:ih/2:0:0,scale={W}:{H},setsar=1[v]", parts[2])

    card([("3 neural networks + 3D geometry", 620, 58, True, "white"),
          ("+ tracking, on a laptop GPU", 700, 58, True, "white"),
          ("~17 FPS real time", 860, 52, False, "0x4FC3F7"),
          ("0 false alarms on empty streets", 940, 52, False, "0x4FC3F7"),
          ("speaks only when it matters", 1020, 52, False, "0x4FC3F7"),
          (REPO_URL, 1250, 36, False, "0xB0BEC5")], 4, parts[3])

    concat = tmp / "list.txt"
    concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    out = OUT_DIR / "linkedin_demo.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(out)],
                   check=True)
    print("Saved", out)


if __name__ == "__main__":
    main()
