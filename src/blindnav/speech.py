"""Turning alerts into sound: voice (text-to-speech) + a beep that comes from the obstacle's side.

Two outputs:
- LiveSpeaker: plays immediately (webcam / phone, real time).
- SoundtrackRecorder: for a video file, places every message at its exact video time and
  muxes the audio into the output .mp4 (our processing is slower than real time, so playing
  live would be out of sync with the video).
"""

import os
import queue
import subprocess
import tempfile
import threading
import wave

import numpy as np

from blindnav.announcer import URGENT, Alert

SAMPLE_RATE = 22050  # Hz
VOICES = {"en": "English", "fr": "French"}
SPEECH_RATE = 2      # Windows scale -10..10 (0 = normal): a bit fast, blind users often prefer it
SAPI_22KHZ_16BIT_MONO = 22
SAPI_CREATE_FOR_WRITE = 3


class Synthesizer:
    """Text -> audio samples, with a cache (the same messages come back often).

    Uses the Windows speech engine (SAPI) directly. pyttsx3 wraps the same engine but
    hangs forever on its second runAndWait() call, so we skip the wrapper.
    """

    def __init__(self, lang: str):
        import comtypes.client  # created here: COM objects must live in the thread that uses them
        self.client = comtypes.client
        self.voice = self.client.CreateObject("SAPI.SpVoice")
        for token in self.voice.GetVoices():
            if VOICES[lang] in token.GetDescription():
                self.voice.Voice = token
        self.voice.Rate = SPEECH_RATE
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text not in self.cache:
            path = os.path.join(tempfile.gettempdir(), "blindnav_tts.wav")
            audio_format = self.client.CreateObject("SAPI.SpAudioFormat")
            audio_format.Type = SAPI_22KHZ_16BIT_MONO
            stream = self.client.CreateObject("SAPI.SpFileStream")
            stream.Format = audio_format
            stream.Open(path, SAPI_CREATE_FOR_WRITE)
            self.voice.AudioOutputStream = stream  # speak into the file instead of the speakers
            self.voice.Speak(text)
            stream.Close()
            with wave.open(path) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768
            loud = np.flatnonzero(np.abs(audio) > 0.01)  # trim the silence around the words
            self.cache[text] = audio[loud[0]:loud[-1]] if loud.size else audio
        return self.cache[text]


def pan_for(clock: int) -> float:
    """Clock position -> stereo pan: -1 = full left, 0 = center, +1 = full right."""
    hours = clock if clock <= 6 else clock - 12  # 10 o'clock -> -2, 2 o'clock -> +2
    return float(np.sin(np.radians(hours * 30)))


def stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    # Constant-power panning: the sound keeps the same loudness wherever it is placed.
    angle = (pan + 1) * np.pi / 4  # 0 (left) .. pi/2 (right)
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)], axis=1)


def beep(urgent: bool) -> np.ndarray:
    """Short tone: double high beep for URGENT, single soft beep for WARNING."""
    freq, n_beeps = (1200, 2) if urgent else (700, 1)
    t = np.arange(int(0.09 * SAMPLE_RATE)) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * freq * t) * np.hanning(t.size)  # hanning = no click
    gap = np.zeros(int(0.05 * SAMPLE_RATE))
    return np.concatenate([tone, gap] * n_beeps)


def render(alert: Alert, text: str, synth: Synthesizer) -> np.ndarray:
    """Full stereo clip for one alert: beep from the obstacle's side, then the voice."""
    pan = pan_for(alert.clock)
    voice = synth(text)
    pause = np.zeros((int(0.08 * SAMPLE_RATE), 2), dtype=np.float32)
    # Beep fully panned (localization), voice only slightly (it must stay easy to understand).
    return np.concatenate([stereo(beep(alert.level == URGENT), pan), pause,
                           stereo(voice, 0.3 * pan)]).astype(np.float32)


class LiveSpeaker:
    """Plays alerts in a background thread, so the video loop never waits for the audio."""

    def __init__(self, lang: str):
        self.lang = lang
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, alert: Alert, text: str) -> None:
        import sounddevice as sd
        if alert.level == URGENT:
            sd.stop()  # an urgent alert interrupts whatever is being said
            while not self.queue.empty():
                self.queue.get_nowait()
        self.queue.put((alert, text))

    def _worker(self) -> None:
        import comtypes
        import sounddevice as sd
        comtypes.CoInitialize()  # Windows speech uses COM, which must be started per thread
        synth = Synthesizer(self.lang)
        while True:
            alert, text = self.queue.get()
            sd.play(render(alert, text, synth), SAMPLE_RATE)
            sd.wait()


class SoundtrackRecorder:
    """Collects (video time, alert) pairs, then builds the audio track and muxes it."""

    def __init__(self, lang: str):
        self.lang = lang
        self.events: list[tuple[float, Alert, str]] = []

    def add(self, t: float, alert: Alert, text: str) -> None:
        self.events.append((t, alert, text))

    def mux(self, video_path: str, duration: float, out_path: str) -> None:
        synth = Synthesizer(self.lang)
        track = np.zeros((int(duration * SAMPLE_RATE) + SAMPLE_RATE, 2), dtype=np.float32)
        for i, (t, alert, text) in enumerate(self.events):
            clip = render(alert, text, synth)
            start = int(t * SAMPLE_RATE)
            # A message is cut when the next one starts (that's what interruption sounds like).
            end = int(self.events[i + 1][0] * SAMPLE_RATE) if i + 1 < len(self.events) else len(track)
            clip = clip[:max(0, min(len(clip), end - start, len(track) - start))]
            track[start:start + len(clip)] = clip

        wav_path = os.path.join(tempfile.gettempdir(), "blindnav_track.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())

        import imageio_ffmpeg
        # Re-encode the video to H.264 (plays everywhere: LinkedIn, phones) and add the audio.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", video_path, "-i", wav_path,
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # H.264 needs even sizes
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-c:a", "aac", "-shortest", out_path], check=True)
