"""
Voice identification: names diarization labels by voice instead of guessing.

Speech recognition says *how many* voices there are and where they change; it
does not know whose they are. Asking an LLM to infer names from the words works
when people address each other by name and fails when they do not, and one
wrong guess mislabels the whole meeting. This module answers acoustically: each
label's voice is compared with enrolled voiceprints of the expected speakers.

  voices/alice-en.m4a -> enroll -> voiceprints/alice.en.npy -> identify -> speaker_0 = Alice

Commands:
  transcribe voices enroll     build voiceprints from the clips in voices/
  transcribe voices check      similarity matrix of the enrolled voiceprints

Needs the voice extra (torch and SpeechBrain): uv tool install 'meeting-transcriber[voice]'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .common import (MEDIA_EXTS, VOICEPRINTS as PRINTS, VOICES, Error, apply_verbosity,
                     log, verbosity_flags)

SR = 16_000
WIN, HOP = 6.0, 3.0        # embedding window / stride, seconds
MIN_SEG = 2.0              # ignore diarization runs shorter than this
MAX_CLUSTER_AUDIO = 300.0   # seconds of speech embedded per label, to name it
SPLIT_AUDIO = 1800.0        # ...and when a label turns out to hold two people, since
                            # then the goal is covering every stretch of it
SPLIT_WIN, SPLIT_HOP = 4.0, 2.0   # finer windows for taking a mixed label apart
MIN_PART = 15.0             # seconds: a second voice under one label must be at
MIN_SHARE = 0.20            # least this loud, and hold this share of the label
MODEL = os.environ.get("VOICEID_MODEL", "speechbrain/spkrec-resnet-voxceleb")

# Calibrated on 15 real meetings (see the README). Genuine matches on call audio
# landed at 0.48-0.88; people who were NOT enrolled (clients, interviewers) at
# 0.08-0.27. In the band between, the voice was right about as often as it was
# wrong (0.325 named a client as an enrolled colleague, 0.348 correctly named a
# fragment), so that band is not decided here: it goes to the LLM as a hint and
# is settled on what the person actually says.
MIN_SCORE = 0.45      # accept outright
MIN_MARGIN = 0.10     # ...and only if the runner-up is this far behind
MIN_VOTES = 0.70      # ...and the cluster's windows agree among themselves
UNKNOWN_SCORE = 0.30  # below this: positive evidence of an unenrolled person


# ── audio ────────────────────────────────────────────────────────────────────

def pcm(path: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode [start, end) of any audio file to 16 kHz mono float32."""
    import soundfile as sf

    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        out = Path(t.name)
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(SR), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        out.unlink(missing_ok=True)
        raise Error(f"ffmpeg failed on {path.name}:\n{r.stderr[-800:]}")
    data, _ = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)
    return data


def windows(data: np.ndarray) -> list[np.ndarray]:
    w, h = int(WIN * SR), int(HOP * SR)
    if len(data) < SR:                      # < 1s of audio carries no speaker identity
        return []
    if len(data) <= w:
        return [data]
    return [data[i:i + w] for i in range(0, len(data) - w + 1, h)]


# ── embeddings ───────────────────────────────────────────────────────────────

_ENCODER = None


def encoder():
    global _ENCODER
    if _ENCODER is None:
        import torch  # noqa: F401  (speechbrain needs it loaded)
        from speechbrain.inference.speaker import EncoderClassifier

        cache = Path.home() / ".cache" / "speechbrain" / MODEL.split("/")[-1]
        _ENCODER = EncoderClassifier.from_hparams(source=MODEL, savedir=str(cache))
    return _ENCODER


def embed(chunks: list[np.ndarray]) -> np.ndarray:
    """Chunk embeddings, L2-normalised, one row each."""
    import torch

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    enc = encoder()
    rows = []
    for c in chunks:
        v = enc.encode_batch(torch.from_numpy(c).unsqueeze(0)).squeeze().detach().numpy()
        rows.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(rows).astype(np.float32)


def centroid(rows: np.ndarray) -> np.ndarray:
    m = rows.mean(0)
    return m / (np.linalg.norm(m) + 1e-9)


# ── enrollment ───────────────────────────────────────────────────────────────

def parse_voice_name(stem: str) -> tuple[str, str]:
    """'alice-en' -> ('alice', 'en'); 'alice-en-2' -> ('alice', 'en'): extra clips
    of the same person and language are averaged into one print."""
    parts = stem.lower().split("-")
    if len(parts) < 2:
        raise Error(f"voices/{stem}: name clips '<person>-<lang>', e.g. alice-en.m4a")
    return parts[0], parts[1]


def enroll(force: bool = False) -> None:
    clips = sorted(f for f in VOICES.iterdir()
                   if f.suffix.lower() in MEDIA_EXTS) if VOICES.is_dir() else []
    if not clips:
        raise Error(f"no voice clips in {VOICES}. Add one solo clip per person, "
                    "named <person>-<lang>, e.g. alice-en.m4a")

    groups: dict[tuple[str, str], list[Path]] = {}
    for f in clips:
        groups.setdefault(parse_voice_name(f.stem), []).append(f)

    PRINTS.mkdir(parents=True, exist_ok=True)
    index = read_index()
    for (person, lang), files in sorted(groups.items()):
        key = f"{person}.{lang}"
        sig = ",".join(f"{f.name}:{f.stat().st_size}" for f in files)
        if not force and index.get(key, {}).get("sig") == sig and (PRINTS / f"{key}.npy").exists():
            log.info("  %-14s unchanged", key)
            continue
        rows = []
        secs = 0.0
        for f in files:
            data = pcm(f)
            secs += len(data) / SR
            rows.append(embed(windows(data)))
        rows = np.concatenate([r for r in rows if r.size])
        if not len(rows):
            log.warning("%s: too short to enroll (needs at least 1s of speech), skipped", key)
            continue
        np.save(PRINTS / f"{key}.npy", centroid(rows))
        index[key] = {"person": person, "lang": lang, "sig": sig, "model": MODEL,
                      "sources": [f.name for f in files],
                      "seconds": round(secs, 1), "chunks": int(len(rows))}
        log.info("  %-14s %6.1fs %4d windows  from %s", key, secs, len(rows),
                 ", ".join(f.name for f in files))
    write_index(index)
    log.info("%d voiceprint(s) in %s", len(index), PRINTS)


def read_index() -> dict:
    p = PRINTS / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_index(index: dict) -> None:
    PRINTS.mkdir(parents=True, exist_ok=True)
    (PRINTS / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


def load_prints() -> dict[str, np.ndarray]:
    """{'alice.en': vector} for every enrolled print built with the current model."""
    index = read_index()
    out = {}
    for key, meta in index.items():
        if meta.get("model") != MODEL:
            continue
        f = PRINTS / f"{key}.npy"
        if f.exists():
            out[key] = np.load(f)
    return out


# ── roster ↔ voiceprint identities ───────────────────────────────────────────

def voice_aliases() -> dict[str, list[str]]:
    """config.yaml `voices:`, e.g. {'alice': ['Alice', 'Alice Smith']}. One voiceprint
    can go by several spellings across projects. A person with no entry is known
    by the name in their file name ('alice' matches 'Alice')."""
    from .common import CONFIG
    import yaml

    data = (yaml.safe_load(CONFIG.read_text()) or {}) if CONFIG.exists() else {}
    return {str(p).lower(): [str(n) for n in (names or [])]
            for p, names in (data.get("voices") or {}).items()}


def _people() -> dict[str, dict[str, np.ndarray]]:
    """{'alice': {'alice.en': vec, 'alice.de': vec}} for every enrolled person."""
    by_person: dict[str, dict[str, np.ndarray]] = {}
    for key, vec in load_prints().items():
        by_person.setdefault(key.split(".")[0], {})[key] = vec
    return by_person


def enrolled_names() -> list[str]:
    """Display names of everyone enrolled: the candidates when no roster is given."""
    aliases = voice_aliases()
    return [(aliases.get(p) or [p.capitalize()])[0] for p in sorted(_people())]


def candidate_prints(candidates: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """-> {roster name: {'alice.en': vec, ...}} for the meeting's roster.
    People with no voiceprint simply do not appear."""
    aliases = voice_aliases()
    by_person = _people()
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in candidates:
        for person, prints in by_person.items():
            names = aliases.get(person) or [person]
            if name.casefold() in (n.casefold() for n in names):
                out[name] = prints
                break
    return out


# ── identification ───────────────────────────────────────────────────────────

def cluster_segments(words: list[dict],
                     cap: float = MAX_CLUSTER_AUDIO) -> dict[str, list[tuple[float, float]]]:
    """Contiguous single-speaker runs per label, longest first.

    Only runs of at least MIN_SEG seconds are kept: short ones sit on a speaker
    boundary and their audio is half somebody else's. `cap` bounds how much audio
    per label is embedded: 90s is plenty to recognise a voice, but taking a mixed
    label apart needs to reach every part of it, so that pass raises the cap."""
    runs: list[tuple[str, float, float]] = []
    for w in words:
        if w.get("type") != "word":
            continue
        sid = w.get("speaker_id") or "speaker_0"
        start, end = w.get("start"), w.get("end")
        if start is None or end is None:
            continue
        if runs and runs[-1][0] == sid and start - runs[-1][2] < 1.0:
            runs[-1] = (sid, runs[-1][1], float(end))
        else:
            runs.append((sid, float(start), float(end)))

    out: dict[str, list[tuple[float, float]]] = {}
    for sid, start, end in runs:
        if end - start >= MIN_SEG:
            out.setdefault(sid, []).append((start, end))
    for sid, segs in out.items():
        segs.sort(key=lambda s: s[1] - s[0], reverse=True)
        kept, total = [], 0.0
        for s, e in segs:
            kept.append((s, e))
            total += e - s
            if total >= cap:
                break
        out[sid] = kept
    return out


def cluster_embeddings(audio: Path, words: list[dict],
                       cap: float = MAX_CLUSTER_AUDIO) -> dict[str, list[tuple]]:
    """{speaker_id: [(start, end, window_embeddings), ...]}, kept per segment so a
    label that turns out to hold two voices can still be taken apart."""
    segs = cluster_segments(words, cap)
    out: dict[str, list[tuple]] = {}
    for sid, spans in sorted(segs.items()):
        rows = []
        for s, e in spans:
            r = embed(windows(pcm(audio, s, e)))
            if r.size:
                rows.append((s, e, r))
        if rows:
            out[sid] = rows
    return out


def assign(scores: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Exclusive greedy assignment: take the best (cluster, name) cell, strike out
    that name, repeat. Two clusters can still share a name only when diarization
    split one voice, handled by re-offering a name once every cluster has had a
    shot at a fresh one.

    Returns {cluster: {name, score, runner_up, margin, accepted}}.
    """
    result: dict[str, dict] = {}
    free = {n for c in scores for n in scores[c]}
    pending = set(scores)

    def best_cell(names: set[str]) -> tuple[str, str, float] | None:
        cell = None
        for c in pending:
            for n, v in scores[c].items():
                if n in names and (cell is None or v > cell[2]):
                    cell = (c, n, v)
        return cell

    # round 1: every cluster gets a distinct name (as many as there are names)
    while pending and free:
        cell = best_cell(free)
        if cell is None:
            break
        c, n, v = cell
        others = sorted((v2 for n2, v2 in scores[c].items() if n2 != n), reverse=True)
        result[c] = {"name": n, "score": round(v, 3),
                     "runner_up": round(others[0], 3) if others else None,
                     "margin": round(v - others[0], 3) if others else None}
        pending.discard(c)
        free.discard(n)

    # round 2: leftover clusters (diarization fragments) may reuse a name
    for c in sorted(pending):
        if not scores[c]:
            continue
        n, v = max(scores[c].items(), key=lambda kv: kv[1])
        others = sorted((v2 for n2, v2 in scores[c].items() if n2 != n), reverse=True)
        result[c] = {"name": n, "score": round(v, 3),
                     "runner_up": round(others[0], 3) if others else None,
                     "margin": round(v - others[0], 3) if others else None}

    for c, r in result.items():
        margin = r["margin"]
        best = max(scores[c].values()) if scores[c] else 0.0
        r["accepted"] = bool(r["score"] >= MIN_SCORE and
                             (margin is None or margin >= MIN_MARGIN))
        # nobody enrolled comes close: this is someone we have no voice for
        r["unknown"] = bool(best < UNKNOWN_SCORE)
    return result


def identify(audio: Path, words: list[dict], candidates: list[str]) -> dict:
    """-> {'speaker_0': {'name', 'score', 'margin', 'accepted', 'votes'}, ...}

    A cluster's score against a person is the cosine between the cluster centroid
    and that person's closest voiceprint (a person enrolled in two languages gets
    two shots; cross-language matching is measurably weaker, so same-language
    wins on its own). `votes` is the share of the cluster's 6-second windows that
    independently pick the assigned name: a second opinion that does not depend
    on the centroid.
    """
    prints = candidate_prints(candidates)
    if not prints:
        return {}
    per_segment = cluster_embeddings(audio, words)
    if not per_segment:
        return {}
    embs = {sid: np.concatenate([r for _, _, r in segs])
            for sid, segs in per_segment.items()}

    scores: dict[str, dict[str, float]] = {}
    per_chunk: dict[str, dict[str, np.ndarray]] = {}
    for sid, rows in embs.items():
        log.debug("  %s: %d windows", sid, len(rows))
        c = centroid(rows)
        scores[sid] = {}
        per_chunk[sid] = {}
        for name, vecs in prints.items():
            sims = {k: float(c @ v) for k, v in vecs.items()}
            scores[sid][name] = max(sims.values())
            per_chunk[sid][name] = np.max(np.stack([rows @ v for v in vecs.values()]), axis=0)

    out = assign(scores)
    for sid, r in out.items():
        if len(prints) > 1:      # with a single candidate every window "votes" for them
            stack = np.stack([per_chunk[sid][n] for n in sorted(per_chunk[sid])])
            winners = np.array(sorted(per_chunk[sid]))[stack.argmax(0)]
            r["votes"] = round(float((winners == r["name"]).mean()), 2)
        else:
            r["votes"] = None
        # A cluster whose own windows disagree is not one voice: usually a
        # diarization fragment straddling a speaker change, so its average is a
        # blend of two people and matches neither.
        if r["votes"] is not None and r["votes"] < MIN_VOTES:
            r["accepted"] = False
        if is_mixed(per_chunk[sid]):
            whole = cluster_segments(words, SPLIT_AUDIO).get(sid) or \
                [(s, e) for s, e, _ in per_segment[sid]]
            log.info("  %s holds more than one voice, re-cutting it", sid)
            r["split"] = split_label(audio, whole, prints)
            if r["split"]:
                r["accepted"] = False   # the label is not one person; its parts are
        r["chunks"] = int(len(embs[sid]))
        r["all"] = {n: round(v, 3) for n, v in sorted(scores[sid].items(),
                                                      key=lambda kv: -kv[1])}
    return out


def is_mixed(window_scores: dict[str, np.ndarray]) -> bool:
    """Does one label hold more than one person?

    Two ways it happens, both visible here. Two *enrolled* people merged: the
    windows split into two confident groups. Or someone we have no sample for is
    hiding inside an enrolled person's label (three people, two labels), which
    shows up as a long run of windows matching nobody. The whole-label average
    hides both: it just reports whoever talks most.
    """
    names = list(window_scores)
    stack = np.stack([window_scores[n] for n in names])
    best, who = stack.max(0), stack.argmax(0)
    total = len(best) * HOP
    strong: dict[str, float] = {}
    stranger = 0.0
    for i, b in enumerate(best):
        if b >= MIN_SCORE:
            strong[names[who[i]]] = strong.get(names[who[i]], 0.0) + HOP
        elif b < UNKNOWN_SCORE:
            stranger += HOP

    # Thresholds are relative as well as absolute: every label has *some* windows
    # that match badly (noise, laughter, one word over a cough). A second person
    # is only worth acting on when they hold a real share of the label, not a
    # fixed number of seconds that a long label reaches by accident.
    def loud(seconds: float) -> bool:
        return seconds >= MIN_PART and seconds >= MIN_SHARE * total

    named = [n for n, s in strong.items() if loud(s)]
    return len(named) >= 2 or (bool(named) and loud(stranger))


def split_label(audio: Path, segments: list[tuple[float, float]],
                prints: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Pull the stretches we can positively recognise out of a mixed label.

    Scribe's own runs are useless here: because it merged the speakers, a single
    run happily spans both of them talking. So the label is re-embedded in short
    windows and each window is matched on its own.

    Only windows that clear the same bar as a whole-label match are claimed.
    Everything else (the other person, and any window that is merely hard to
    read) stays under the original label, which is then handed to the LLM marked
    as mixed. Guessing at the remainder is how a transcript ends up confidently
    wrong, which is worse than one label the reader knows to check.
    """
    global WIN, HOP
    keep = (WIN, HOP)
    WIN, HOP = SPLIT_WIN, SPLIT_HOP
    try:
        wins: list[tuple[float, float, np.ndarray]] = []
        for start, end in sorted(segments):     # chronological: spans merge by time
            rows = embed(windows(pcm(audio, start, end)))
            for i in range(len(rows)):
                ws = min(start + i * HOP, end)
                # each window owns its hop slot; the last one owns the tail too,
                # so no stretch of speech falls between the slots
                we = end if i == len(rows) - 1 else min(ws + HOP, end)
                wins.append((ws, we, rows[i]))
    finally:
        WIN, HOP = keep
    if not wins:
        return []

    E = np.stack([w[2] for w in wins])
    names = list(prints)
    stack = np.stack([(E @ np.stack(list(prints[n].values())).T).max(1) for n in names])
    order = np.sort(stack, axis=0)
    top, second = order[-1], (order[-2] if len(names) > 1 else np.zeros_like(order[-1]))
    pick = [names[i] if t >= MIN_SCORE and t - s >= MIN_MARGIN else None
            for i, t, s in zip(stack.argmax(0), top, second)]
    for i in range(1, len(pick) - 1):        # a lone dissenting window is noise
        if pick[i - 1] == pick[i + 1] and pick[i] != pick[i - 1]:
            pick[i] = pick[i - 1]

    parts: list[dict] = []
    for (start, end, _), name in zip(wins, pick):
        if parts and parts[-1]["name"] == name and start - parts[-1]["end"] < 0.5:
            parts[-1]["end"] = round(end, 2)
        else:
            parts.append({"start": round(start, 2), "end": round(end, 2),
                          "name": name})
    return [p for p in parts if p["name"] is not None]


# ── cli ──────────────────────────────────────────────────────────────────────

def cmd_check() -> None:
    prints = load_prints()
    if not prints:
        raise Error("no voiceprints yet: run `transcribe voices enroll`")
    keys = sorted(prints)
    print(f"model: {MODEL}\n")
    print("           " + "".join(f"{k[:10]:>11}" for k in keys))
    for a in keys:
        row = "".join(f"{float(prints[a] @ prints[b]):>11.3f}" for b in keys)
        print(f"{a[:10]:>10} {row}")
    same = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
            if a.split(".")[0] == b.split(".")[0]]
    diff = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
            if a.split(".")[0] != b.split(".")[0]]
    if same:
        print(f"\nsame person, other language: "
              f"{min(float(prints[a] @ prints[b]) for a, b in same):.3f} (min)")
    if diff:
        print(f"different people:            "
              f"{max(float(prints[a] @ prints[b]) for a, b in diff):.3f} (max)")


def run(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="transcribe voices", parents=[verbosity_flags()],
                                 description=f"Voiceprints for speaker identification. "
                                             f"Clips go in {VOICES}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll", parents=[verbosity_flags()],
                       help="build voiceprints from the clips in voices/")
    e.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    sub.add_parser("check", parents=[verbosity_flags()],
                   help="similarity matrix of the enrolled voiceprints")
    args = ap.parse_args(argv)
    apply_verbosity(args)
    if args.cmd == "enroll":
        enroll(args.force)
    else:
        cmd_check()
