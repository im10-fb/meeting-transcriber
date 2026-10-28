"""
`transcribe record`: capture a call on a Mac into the inbox folder.

Records two live streams and mixes them into ONE file:
  - system audio through BlackHole 2ch: the remote participants' voices
  - your microphone: your own voice

BlackHole taps the audio stream before it reaches any physical output, so it
makes no difference whether you listen on speakers or headphones. Your own voice
only exists at the mic, so it is recorded separately and mixed in here.

Why two ffmpeg processes and a mix rather than one? macOS avfoundation deadlocks
when one ffmpeg opens two audio inputs, so each device gets its own capture
process (to a temp WAV) and a final pass mixes them into a single Opus track.

One-time setup (so you hear the call AND BlackHole gets a copy):
  1. brew install blackhole-2ch, then open Audio MIDI Setup.
  2. Create a Multi-Output Device containing BOTH BlackHole 2ch and your
     speakers or headphones (tick drift correction on the non-clock device).
  3. Before a call, set the Mac's sound output to that Multi-Output Device.
  The terminal app needs Microphone permission (System Settings > Privacy &
  Security > Microphone); macOS asks on the first run.

Press Ctrl-C to stop; the streams are finalized, mixed and left in the inbox.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .common import (Error, apply_verbosity, load_config, log, require, settings_for,
                     verbosity_flags)

DEFAULT_SYSTEM = "BlackHole 2ch"


def list_devices() -> list[tuple[int, str]]:
    """Parse ffmpeg's avfoundation audio device list -> [(index, name), ...]."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if in_audio:
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
            elif "AVFoundation" not in line:
                break  # left the audio-devices block
    return devices


def resolve(spec: str, devices: list[tuple[int, str]], what: str) -> int:
    """Turn a device name (substring, case-insensitive) or index into an index."""
    if spec.isdigit():
        idx = int(spec)
        if any(i == idx for i, _ in devices):
            return idx
        raise Error(f"no audio input at index {idx} (see `transcribe record --list`)")
    matches = [i for i, name in devices if spec.lower() in name.lower()]
    if not matches:
        avail = ", ".join(f"[{i}] {n}" for i, n in devices)
        raise Error(f"{what} device not found: {spec!r}. Available: {avail}")
    if len(matches) > 1:
        raise Error(f"{what} device {spec!r} is ambiguous (matches indices {matches})")
    return matches[0]


def capture_proc(idx: int, out: Path) -> subprocess.Popen:
    """One avfoundation device -> a temp WAV. Own session so the terminal's
    Ctrl-C doesn't hit it; a clean SIGINT is forwarded so ffmpeg finalizes the file."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "avfoundation", "-i", f":{idx}",
           "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    return subprocess.Popen(cmd, start_new_session=True)


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def mean_volume(path: Path) -> float | None:
    """Mean loudness in dBFS via ffmpeg volumedetect (None if unreadable).
    Near-silence reads about -70 dB or lower; speech typically -15 to -35 dB."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", r.stderr)
    return float(m.group(1)) if m else None


def report_level(label: str, path: Path) -> None:
    mv = mean_volume(path)
    if mv is None:
        log.info("  %s: level unknown", label)
    elif mv <= -70:
        log.warning("%s: %.0f dB, looks SILENT: nothing was captured there", label, mv)
    else:
        log.info("  %s: %.0f dB, has audio", label, mv)


def ticker(started: float, stop_evt: threading.Event) -> None:
    while not stop_evt.wait(0.25):
        el = int(time.monotonic() - started)
        sys.stderr.write(f"\rRecording {el // 60:02d}:{el % 60:02d}  (Ctrl-C to stop)")
        sys.stderr.flush()


def run(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="transcribe record", parents=[verbosity_flags()],
                                 description="Record a call on macOS (system audio via "
                                             "BlackHole plus your mic) into the inbox.")
    ap.add_argument("--list", action="store_true", help="list audio input devices and exit")
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help=f"system-audio device name or index (default: {DEFAULT_SYSTEM!r})")
    ap.add_argument("--mic", help="microphone name or index (default: the first input "
                                  "that is not the system device)")
    ap.add_argument("--no-mic", action="store_true",
                    help="capture only system audio, not your microphone")
    ap.add_argument("--keep-parts", action="store_true",
                    help="also keep the separate system and mic tracks, to check each "
                         "source landed")
    ap.add_argument("-o", "--out", help="output file (default: <inbox>/<date>-<time>.ogg)")
    args = ap.parse_args(argv)
    apply_verbosity(args)

    if sys.platform != "darwin":
        raise Error("recording is macOS only (it uses avfoundation and BlackHole). "
                    "Record with any app and pass the file to transcribe.")
    require("ffmpeg")
    devices = list_devices()
    if args.list:
        if not devices:
            raise Error("no avfoundation audio devices found")
        print("Audio input devices:")
        for i, name in devices:
            print(f"  [{i}] {name}")
        return
    if not devices:
        raise Error("no avfoundation audio devices found (is ffmpeg built with avfoundation?)")

    system_idx = resolve(args.system, devices, "system-audio")
    mic_idx = None
    if not args.no_mic:
        if args.mic:
            mic_idx = resolve(args.mic, devices, "microphone")
        else:
            mic_idx = next((i for i, n in devices if i != system_idx
                            and "blackhole" not in n.lower()), None)
            if mic_idx is None:
                raise Error("no microphone found; pass --mic or --no-mic (see --list)")
    if system_idx == mic_idx:
        raise Error("system and mic are the same device: pass distinct --system and --mic")

    names = dict(devices)
    if args.out:
        out = Path(args.out).expanduser()
    else:
        inbox = Path(settings_for(load_config(), None)["inbox"]).expanduser()
        out = inbox / f"{dt.datetime.now():%Y-%m-%d-%H%M%S}.ogg"
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("System audio: [%d] %s", system_idx, names[system_idx])
    log.info("Microphone:   %s", f"[{mic_idx}] {names[mic_idx]}" if mic_idx is not None
             else "off (system audio only)")
    if "blackhole" not in names[system_idx].lower():
        log.warning("the system device is not BlackHole, so this may record a mic rather "
                    "than the call. Route the Mac's output to a Multi-Output Device that "
                    "includes BlackHole 2ch (see `transcribe record --help`).")
    log.info("Saving to %s", out)

    tmp = Path(tempfile.mkdtemp(prefix="rec_"))
    parts: list[tuple[str, Path]] = []  # (label, wav)
    procs: list[subprocess.Popen] = []
    try:
        sys_wav = tmp / "system.wav"
        procs.append(capture_proc(system_idx, sys_wav))
        parts.append(("system", sys_wav))
        if mic_idx is not None:
            mic_wav = tmp / "mic.wav"
            procs.append(capture_proc(mic_idx, mic_wav))
            parts.append(("mic", mic_wav))

        # a capture that dies at once means a bad device or no permission
        time.sleep(0.6)
        for p, (label, _) in zip(procs, parts):
            if p.poll() is not None:
                raise Error(f"the {label} capture exited immediately (code {p.returncode}): "
                            "check the device routing and Microphone permission")

        stop_evt = threading.Event()
        started = time.monotonic()
        t = threading.Thread(target=ticker, args=(started, stop_evt), daemon=True)
        t.start()
        try:
            while any(p.poll() is None for p in procs):
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass
        finally:
            stop_evt.set()
            t.join(timeout=1)
        secs = int(time.monotonic() - started)
        sys.stderr.write("\r" + " " * 40 + "\r")
        log.info("Stopped after %02d:%02d, finalizing", secs // 60, secs % 60)
        for p in procs:
            stop(p)

        good = [(label, p) for label, p in parts if p.exists() and p.stat().st_size > 1024]
        if not good:
            raise Error("no audio captured: check the device routing and Microphone "
                        "permission")

        # per-source levels: the fastest way to confirm your own voice landed
        log.info("Captured levels:")
        for label, p in good:
            report_level("mic (your voice)" if label == "mic" else "system (remote)", p)
        if mic_idx is not None and not any(label == "mic" for label, _ in good):
            log.warning("the mic track is missing: your voice was NOT recorded")

        mix = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        for _, p in good:
            mix += ["-i", str(p)]
        if len(good) > 1:
            # normalize=0 keeps each source at full level, so quiet remote audio is
            # not buried under your own mic
            mix += ["-filter_complex",
                    f"amix=inputs={len(good)}:duration=longest:normalize=0[a]",
                    "-map", "[a]"]
        mix += ["-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(out)]
        r = subprocess.run(mix, capture_output=True, text=True)
        if r.returncode != 0:
            raise Error(f"the final mix failed:\n{r.stderr[-1500:]}")

        kept: list[Path] = []
        if args.keep_parts:
            # NOT next to the mix: the inbox picks up every audio file, and would
            # transcribe the call together with its own two source tracks
            parts_dir = out.parent / "parts"
            parts_dir.mkdir(exist_ok=True)
            for label, p in good:
                dest = parts_dir / f"{out.stem}.{label}.wav"
                shutil.copy2(p, dest)
                kept.append(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    log.info("Saved %.1f MB. Next: transcribe%s", out.stat().st_size / 1e6,
             f" {out}" if args.out else "")
    print(f"{'recording':<11}{out}")
    for k in kept:
        print(f"{'part':<11}{k}")
