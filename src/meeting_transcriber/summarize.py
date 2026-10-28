"""
Transcript -> two summaries, written in the conversation's language by default.

  summary.md         one LLM call works out what kind of conversation this was and
                     writes only the sections that carry real content (TL;DR,
                     decisions, next steps, open questions, ...) under a YAML
                     frontmatter (title / type / participants / tldr / tags).
  summary-short.txt  a second call boils that down to plain text sized for a chat
                     message (what was agreed, then action steps per person).
                     Printed in full at the end of the run, ready to paste.

Runs after every transcript, or on its own: `transcribe summarize <transcript>`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from . import llm
from .common import (Error, apply_verbosity, find_project, load_config, log,
                     settings_for, verbosity_flags)

SUMMARY_CHAR_CAP = 400_000  # cap on transcript text sent to the LLM
SHORT_CHAR_CAP = 1500       # target length of the short version, set in the prompt


# ── prompts ──────────────────────────────────────────────────────────────────
# Both prompts are written to be neutral: the summary reports what was said by
# whom, it does not take sides, grade anyone, or fill gaps from the project
# description. The project context is there for spelling and roles only.

SUMMARY_PROMPT = """You turn the transcript of a recorded conversation into a structured summary.

Project: {name}
About the project (only to understand terms and roles; it is not a source of facts
about this conversation): {context}
Speakers in the transcript: {speakers}
{roles}Date: {date}

The transcript is made of lines "Name: text". A name followed by "?" (for example
"Anna?:") means the speaker was identified with low confidence: do not attribute
decisions or commitments to that person as if it were certain.

STEP 1. From the content, work out what kind of conversation this is (for example:
client meeting, internal sync, technical discussion, planning, job interview,
lesson, consultation, doctor's appointment, personal conversation). Do not force a
business format onto a conversation that is not one.

STEP 2. Write the summary in Markdown. First a YAML frontmatter:
---
title: <short title>
date: {date}
type: <kind of conversation, a word or short phrase>
participants: [<names of the people who actually spoke>]
tldr: <one sentence: the main outcome>
tags: [<3 to 6 tags, comma separated>]
---

Then the body. Include ONLY sections with real content (headings with `## `):
- `## TL;DR`: 2 to 4 sentences (always)
- `## Context`: why the conversation happened, if it is not obvious
- `## Key points`: the substance, as bullets or short paragraphs
- `## Decisions`: only what was explicitly agreed
- `## Next steps`: a table `| Action | Owner | Due |`, only if tasks came up. Fill in
  the owner only when the conversation makes clear who does it, otherwise leave the
  cell empty. Only due dates that were actually named.
- `## Open questions`: what is unresolved or still to be clarified
- `## Worth remembering`: facts useful months from now (numbers, agreements, the
  reasons behind decisions, facts about people and companies that came up)
- `## Numbers`: key figures, if any were mentioned
- Sections that suit the kind of conversation, if they come naturally (for a job
  interview: the tasks and questions and how they were answered; for a lesson: what
  was covered; for a technical discussion: the architecture decisions).

Rules:
- Write EVERYTHING in {language}: the text, the section headings, the table header
  and the value of `type`. The section names above are in English; translate them
  if the output language is different. The frontmatter keys (title, date, type,
  participants, tldr, tags) stay as they are.
- Use ONLY what was said. Do not invent reasons, deadlines, owners or outcomes. No
  content, no section.
- Be neutral towards every participant: give each side's position in equal detail,
  without praise or judgement. Do not assess who was right or how anyone did unless
  that was said in the conversation; attribute any such assessment to whoever made it.
- Where opinions differed or a question stayed open, say so; do not pick a side.
- Keep important details (names, product names, numbers, the wording of
  agreements), but do not retell the conversation line by line.
- No LaTeX (`$...$`, `\\rightarrow`): write numbers, formulas and arrows as plain text.
- Reply with ONLY the Markdown document (frontmatter and body), no ``` fences and no
  text before or after it.

Transcript:
{transcript}
"""

SHORT_PROMPT = """You condense a detailed conversation summary into a short message that will
be sent in a chat app and read on a phone.

Project: {name}
Participants: {speakers}

Reply format: PLAIN TEXT, no Markdown (no #, *, **, `, tables or links).

Summary

<One dense paragraph, 3 to 5 sentences: what was concluded and why. Not a list of
topics but the substance: what was chosen, what was rejected, what is still open.>
• <a secondary but important point, one line>
• <another, if there is one; at most 4 bullets, none is fine>

Action steps

<Name>
• <what this person does: concrete, with details>
• <another task of this person>

<Next name>
• <task>

Rules:
- Language: {language}. Translate the two headings "Summary" and "Action steps"
  into that language if it is not English, and put a blank line after each.
- Every bullet starts with the character • and a space. The paragraph under the
  first heading has no bullet.
- A heading inside the action steps is exactly one name, spelled as in the summary.
- 5 to 10 times shorter than the summary, {cap} characters at most in total.
- Only outcomes and tasks. No sections about facts, numbers or tags, and no retelling
  of how the conversation went.
- Only tasks that have an owner in the summary. Each person appears once. A shared
  task goes once, under whoever leads it, with the other person named in the line.
  Put the due date in the line if one was named.
- Do not repeat in the tasks what the paragraph already says.
- Be neutral: no judgement of participants, no conclusions that are not in the summary.
- If there are no tasks, leave out the action steps block entirely.
- Reply with only the text of the message, no explanation and no ``` fences.

Summary:
{summary}
"""


def language_name(language: str | None) -> str:
    """'auto' (the default) asks for the conversation's own language."""
    if not language or str(language).lower() == "auto":
        return ("the language most of the conversation is held in (if it mixes "
                "languages, the one most of the speech is in)")
    return str(language)


def _clip(text: str) -> str:
    if len(text) <= SUMMARY_CHAR_CAP:
        return text
    head = text[: SUMMARY_CHAR_CAP * 3 // 4]
    tail = text[-SUMMARY_CHAR_CAP // 4:]
    return f"{head}\n...\n[transcript shortened]\n...\n{tail}"


def split_frontmatter(md: str) -> tuple[dict, str]:
    """-> (frontmatter dict, body). Tolerant of a missing or broken block."""
    import yaml

    if md.startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md, re.DOTALL)
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                fm = {}
            return (fm if isinstance(fm, dict) else {}), m.group(2)
    return {}, md


def speakers_in(transcript: str) -> list[str]:
    """Who actually talks, in order of first line, `?` marks dropped. Taken from
    the transcript rather than the roster: the roster can list people who were
    not on this call and miss guests who were."""
    _, body = split_frontmatter(transcript)
    names: list[str] = []
    for m in re.finditer(r"^([^\n:]{1,40}?)\??: ", body, re.M):
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


# ── generation ───────────────────────────────────────────────────────────────

def generate_summary(transcript: str, settings: dict, llm_cfg: dict, date: str,
                     language: str | None = None, model: str | None = None) -> str:
    speakers = speakers_in(transcript)
    roles = {n: r for n, r in (settings.get("roles") or {}).items() if n in speakers}
    roles_block = ""
    if roles:
        roles_block = "Roles:\n" + "\n".join(f"- {n}: {d}" for n, d in roles.items()) + "\n"
    prompt = SUMMARY_PROMPT.format(
        name=settings.get("name") or "-",
        context=" ".join((settings.get("context") or "-").split()),
        speakers=", ".join(speakers) or "-",
        roles=roles_block,
        date=date or "",
        language=language_name(language),
        transcript=_clip(transcript),
    )
    log.info("Writing the summary with %s", model or llm_cfg.get("model"))
    md = llm.complete(llm_cfg, prompt, model).strip()
    md = re.sub(r"^```(?:markdown)?\s*\n", "", md)   # an accidental fence
    md = re.sub(r"\n```\s*$", "", md)
    if not md:
        raise Error("summary: the LLM returned an empty reply")
    return md.strip() + "\n"


def _plainify(text: str) -> str:
    """Chat apps show Markdown syntax literally, so it has to go."""
    text = re.sub(r"^```.*$", "", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def generate_short(summary_md: str, settings: dict, llm_cfg: dict,
                   language: str | None = None, model: str | None = None) -> str:
    fm, _ = split_frontmatter(summary_md)
    people = fm.get("participants") or []
    prompt = SHORT_PROMPT.format(
        name=settings.get("name") or "-",
        speakers=", ".join(map(str, people)) if isinstance(people, list) else people or "-",
        # the long summary is already in the right language: follow it
        language=language_name(language) if language and language != "auto"
        else "the language of the summary below",
        cap=SHORT_CHAR_CAP,
        summary=summary_md,
    )
    log.info("Writing the short version")
    text = llm.complete(llm_cfg, prompt, model)
    if not text.strip():
        raise Error("short summary: the LLM returned an empty reply")
    return _plainify(text)


def show_short(text: str) -> None:
    """Printed whole on stdout: copying from the terminal is how it gets into chat."""
    rule = "-" * 52
    print(f"\n{rule}\n{text}{rule}\n")


# ── file layout ──────────────────────────────────────────────────────────────

def summary_target(transcript_path: Path) -> tuple[Path, str]:
    """Where the summaries go, and under what stem.

    A meeting folder holds everything about one meeting, so the summaries land
    next to the transcript: <meeting>/transcript.md -> summary.md. Flat folders
    file transcripts by stem next to other notes and keep summaries in summaries/.
    """
    if transcript_path.name == "transcript.md":
        return transcript_path.parent, "summary"
    return transcript_path.parent / "summaries", transcript_path.stem


def short_path(long_path: Path) -> Path:
    """summary.md -> summary-short.txt, beside it."""
    return long_path.with_name(f"{long_path.stem}-short.txt")


def summarize_transcript(transcript_md: str, transcript_path: Path, settings: dict,
                         llm_cfg: dict, date: str = "", language: str | None = None,
                         model: str | None = None, show: bool = True
                         ) -> tuple[Path, Path | None]:
    """Transcript -> summary.md and summary-short.txt, written over any old ones:
    a re-run is a newer version of the same meeting."""
    md = generate_summary(transcript_md, settings, llm_cfg, date, language, model)
    out_dir, stem = summary_target(transcript_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_out = out_dir / f"{stem}.md"
    long_out.write_text(md, encoding="utf-8")
    # best effort: the long summary is on disk, a failure here loses nothing
    try:
        text = generate_short(md, settings, llm_cfg, language, model)
    except Error as e:
        log.warning("short version skipped: %s", e)
        return long_out, None
    short_out = short_path(long_out)
    short_out.write_text(text, encoding="utf-8")
    if show:
        show_short(text)
    return long_out, short_out


# ── `transcribe summarize` ───────────────────────────────────────────────────

def run(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="transcribe summarize", parents=[verbosity_flags()],
                                 description="(Re)write the summaries of a transcript.")
    ap.add_argument("transcript", help="a transcript .md written by transcribe")
    ap.add_argument("-p", "--project", help="project from config.yaml (context and roles)")
    ap.add_argument("--date", default="", help="meeting date YYYY-MM-DD (default: from the path)")
    ap.add_argument("--language", metavar="LANG",
                    help="language of the summaries (default: the conversation's)")
    ap.add_argument("--model", help="use this model instead of llm.model")
    ap.add_argument("--short-only", action="store_true",
                    help="the file is a summary.md: only rewrite its short version")
    args = ap.parse_args(argv)
    apply_verbosity(args)

    tpath = Path(args.transcript).expanduser().resolve()
    if not tpath.is_file():
        raise Error(f"file not found: {tpath}")
    text = tpath.read_text(encoding="utf-8")
    cfg = load_config()
    project = find_project(cfg.get("projects") or {}, args.project)[1] if args.project \
        else None
    settings = settings_for(cfg, project)
    llm_cfg = cfg.get("llm") or {}
    language = args.language or settings["summary_language"]

    if args.short_only:
        short = generate_short(text, settings, llm_cfg, language, args.model)
        out = short_path(tpath)
        out.write_text(short, encoding="utf-8")
        show_short(short)
        print(f"{'short':<11}{out}")
        return

    # the date: the flag, else a YYYY-MM-DD in the file name or its folder's name
    date = args.date
    if not date:
        for name in (tpath.stem, tpath.parent.name):
            if m := re.match(r"(\d{4}-\d{2}-\d{2})", name):
                date = m.group(1)
                break

    long_out, short_out = summarize_transcript(text, tpath, settings, llm_cfg, date,
                                               language, args.model)
    print(f"{'summary':<11}{long_out}")
    if short_out:
        print(f"{'short':<11}{short_out}")
