#!/usr/bin/env python3
"""Core logic for converting line-by-line LRC files into word-by-word (enhanced) LRC.

The conversion does not transcribe anything.  It takes the lyric text you already
have, and for every timed line it force-aligns that text against the slice of the
audio that belongs to the line (from the line's own timestamp up to the next
line's timestamp).  The result is the same text with an inline ``<mm:ss.xx>`` tag
in front of every word:

    [00:28.90] I got a feeling
    ->
    [00:28.90]<00:28.90>I <00:29.34>got <00:29.62>a <00:29.81>feeling

Files that are already word-by-word are detected and skipped.  Metadata tags,
blank lines, instrumental markers (a timestamp with no text) and lines that
already carry word tags are copied through untouched.

This module has no GUI code and only imports torch / whisper lazily inside
``WhisperLineAligner`` so the parsing and formatting can be unit tested without
the heavy dependencies installed.
"""
from __future__ import annotations

import bisect
import contextlib
import io
import os
import re
import shutil
import subprocess
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__version__ = "1.1.0"

# Name written into the ``[re:...]`` provenance tag of every converted file.
WRITER = "lrc-align"

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".aiff", ".aif", ".alac", ".wv", ".ape", ".mp4",
}

SAMPLE_RATE = 16000

# Longest audio window used for a single line.  Whisper works on 30 second
# chunks, and a lyric line never needs more than that.
MAX_LINE_WINDOW = 30.0
# Audio included before the first line of a slice so a clipped consonant is not lost.
LEAD_PAD = 0.15
# Lead used when a line has no previous line to serve as context.
LEAD_PAD_SOLO = 0.5
# Neighbouring lines are aligned together with a line as context when their
# tags are this close; audio after the next line's tag included as context.
CONTEXT_GAP = 12.0
CONTEXT_TAIL = 6.0
# Audio after a line's tag when no next line is close enough to be context.
SOLO_TAIL_MAX = 15.0
# A line's words never spread over more than this: base + per-word seconds
# from the first word's start to the last word's start.  In 251 professionally
# timed lines across four songs none exceeded 1.0 + 0.4 s per word; the extra
# allowance is for slow ballads.
LINE_SPAN_BASE = 1.0
LINE_SPAN_PER_WORD = 0.5
# ...but slow ballads sing a word per second, so a line may also span up to
# this many times the song's own typical seconds-per-word (median over lines).
SPAN_PACE_FACTOR = 2.5
# No single gap between two words of a line may exceed this many times the
# song's pace (or the floor): a held note before an instrumental break tends to
# drag the last word into the break.  Professional files show mid-line pauses
# up to about 4x the song's pace, never more.
GAP_PACE_FACTOR = 4.0
GAP_FLOOR = 2.5
# Mean aligner word confidence below which a file is flagged in the report
# (loud or screamed vocals the model cannot follow well).
LOW_CONFIDENCE = 0.5


def line_span_cap(n_words: int) -> float:
    return LINE_SPAN_BASE + LINE_SPAN_PER_WORD * max(1, n_words)


def song_pace(raw_times: Iterable[Sequence[Optional[float]]]) -> Optional[float]:
    """Median seconds per word across a song's aligned lines."""
    paces = []
    for times in raw_times:
        known = [t for t in times if t is not None]
        if len(known) >= 3 and not alignment_is_poor(times):
            paces.append((known[-1] - known[0]) / (len(known) - 1))
    if len(paces) < 3:      # too little to know the song's pace
        return None
    paces.sort()
    n = len(paces)
    return paces[n // 2] if n % 2 else (paces[n // 2 - 1] + paces[n // 2]) / 2


def cap_line_span(times: List[float], n_words: Optional[int] = None, pace: Optional[float] = None) -> List[float]:
    """Compress a line's word times proportionally when they spread too far."""
    if len(times) < 2:
        return times
    n = n_words or len(times)
    cap = line_span_cap(n)
    if pace:
        cap = max(cap, LINE_SPAN_BASE + SPAN_PACE_FACTOR * pace * (n - 1))
    # Close single gaps that are far longer than the song ever pauses mid-line.
    gap_cap = max(GAP_FLOOR, GAP_PACE_FACTOR * pace) if pace else GAP_FLOOR
    times = list(times)
    for k in range(1, len(times)):
        excess = (times[k] - times[k - 1]) - gap_cap
        if excess > 0:
            for j in range(k, len(times)):
                times[j] -= excess
    span = times[-1] - times[0]
    if span <= cap:
        return times
    t0 = times[0]
    scale = cap / span
    return [t0 + (t - t0) * scale for t in times]
# When re-timing line tags to the sung first word, never move a tag earlier than
# this or later than this relative to the original tag.  Lyric files tend to be
# tagged a little early, while the aligner hears long sustained first notes a
# little late, so the bounds are deliberately tight.
MAX_EARLY_SHIFT = 0.3
MAX_LATE_SHIFT = 0.35
# Measured against a professionally timed file, the aligner hears the first
# word of a line this much later than it really starts.
FIRST_WORD_BIAS = 0.12
# Tags already this close to the sung first word are left exactly as they are.
# Lines are only ever moved LATER: lyric files run early far more often than
# late, and an aligner estimate well before the tag is almost always the
# aligner grabbing an earlier phrase.
RETIME_DEADBAND = 0.15
# Per-line re-timing only happens when the song as a whole looks early
# (median shift at least this much); a few late-heard lines in an otherwise
# accurate file are aligner noise, not tag errors.
RETIME_GATE = 0.15
# When nearly every line is off by the same amount the lyric file was timed to
# a different edit of the song: shift the whole file by that amount instead.
GLOBAL_OFFSET_MAX_SPREAD = 0.3     # interquartile range of per-line shifts
GLOBAL_OFFSET_MIN_LINES = 6
MAX_GLOBAL_OFFSET = 10.0
# Room for the previous line's tail.  When a line's last aligned word lands at
# or after the next line's tag while the next line's own first word is heard
# later still, that tag is early and the last word would be shown for an
# instant before the player flips lines.  Move the tag to just after the tail,
# never past the next line's first word and never more than this.
TAIL_ROOM = 0.15
TAIL_ROOM_TRIGGER = 0.05
MAX_TAIL_SHIFT = 0.6
# Audio included after the next line's timestamp so ad-libs that overlap the
# next line can still be placed where they are sung.
TAIL_PAD = 0.5
# Minimum spacing enforced between consecutive word tags.
MIN_WORD_STEP = 0.01
# Rough spoken/sung duration per word used when alignment fails for a line.
FALLBACK_SECONDS_PER_WORD = 0.45

LINE_TS_RE = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")
WORD_TS_RE = re.compile(r"<(\d+):(\d{1,2})(?:[.:](\d{1,3}))?>")
# ID tags such as [ar:Artist], [ti:Title], [offset:+200]
META_RE = re.compile(r"^\s*\[([A-Za-z#][^\]:]*):([^\]]*)\]\s*$")
# The standard "program that created this file" id tag.  Beetdrop reads it as
# ``writer version source timing`` and only looks at the first one.
PROVENANCE_RE = re.compile(r"^\s*\[re:([^\]]*)\]\s*$")
ZIP_ESCAPE_RE = re.compile(r"#U([0-9A-Fa-f]{4})")

# An alignment function receives the absolute window (seconds) inside the current
# song and the text of one line.  It returns (word, start, end, probability)
# tuples with absolute times.  Returning an empty list means "could not align".
AlignFn = Callable[[float, float, str], List[Tuple[str, float, float, float]]]


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def parse_timestamp(minutes: str, seconds: str, fraction: Optional[str]) -> float:
    value = int(minutes) * 60 + int(seconds)
    if fraction:
        value += int(fraction) / (10 ** len(fraction))
    return value


def format_timestamp(seconds: float, decimals: int = 2) -> str:
    """Format seconds as ``mm:ss.xx`` (truncating, never rounding past the tag)."""
    seconds = max(0.0, seconds)
    scale = 10 ** decimals
    total = int(seconds * scale + 1e-6)
    minutes, rem = divmod(total, 60 * scale)
    secs, frac = divmod(rem, scale)
    return f"{minutes:02d}:{secs:02d}.{frac:0{decimals}d}"


# --------------------------------------------------------------------------- #
# LRC parsing
# --------------------------------------------------------------------------- #
@dataclass
class LrcLine:
    raw: str
    start: Optional[float] = None       # None for metadata / blank / untimed lines
    tag: str = ""                       # the original ``[mm:ss.xx]`` text
    text: str = ""                      # lyric text after the timestamp(s)
    has_word_tags: bool = False
    extra_tags: List[Tuple[float, str]] = field(default_factory=list)

    @property
    def is_lyric(self) -> bool:
        return self.start is not None and bool(self.text.strip())

    @property
    def needs_alignment(self) -> bool:
        return self.is_lyric and not self.has_word_tags


def parse_lrc(content: str) -> List[LrcLine]:
    """Parse LRC text into lines.  Lines with several timestamps are expanded."""
    lines: List[LrcLine] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or META_RE.match(stripped):
            lines.append(LrcLine(raw=raw))
            continue

        tags: List[Tuple[float, str]] = []
        pos = 0
        while True:
            m = LINE_TS_RE.match(stripped, pos)
            if not m:
                break
            tags.append((parse_timestamp(m.group(1), m.group(2), m.group(3)), m.group(0)))
            pos = m.end()

        if not tags:
            lines.append(LrcLine(raw=raw))
            continue

        text = stripped[pos:].strip()
        has_words = bool(WORD_TS_RE.search(text))
        first_start, first_tag = tags[0]
        lines.append(LrcLine(raw=raw, start=first_start, tag=first_tag, text=text,
                             has_word_tags=has_words, extra_tags=tags[1:]))
    return expand_multi_timestamp_lines(lines)


def expand_multi_timestamp_lines(lines: List[LrcLine]) -> List[LrcLine]:
    """``[00:10.00][01:20.00]text`` becomes two lines so each gets its own window."""
    out: List[LrcLine] = []
    for line in lines:
        if not line.extra_tags:
            out.append(line)
            continue
        out.append(LrcLine(raw=f"{line.tag}{line.text}", start=line.start, tag=line.tag,
                           text=line.text, has_word_tags=line.has_word_tags))
        for start, tag in line.extra_tags:
            out.append(LrcLine(raw=f"{tag}{line.text}", start=start, tag=tag,
                               text=line.text, has_word_tags=line.has_word_tags))
    return out


def is_word_level(lines: Sequence[LrcLine]) -> bool:
    """True when every lyric line already has inline word tags (nothing to do)."""
    lyric_lines = [ln for ln in lines if ln.is_lyric]
    return bool(lyric_lines) and all(ln.has_word_tags for ln in lyric_lines)


def has_any_word_tags(lines: Sequence[LrcLine]) -> bool:
    return any(ln.has_word_tags for ln in lines if ln.is_lyric)


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Word timing
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> List[str]:
    """Split a lyric line into the tokens that will each receive a tag."""
    return text.split()


def _squash(word: str) -> str:
    return re.sub(r"\s+", "", word)


def map_aligned_words_to_tokens(
    tokens: Sequence[str],
    aligned: Sequence[Tuple[str, float, float, float]],
) -> List[Optional[float]]:
    """Give every token the start time of the aligned word that covers it.

    The aligner may split words differently than ``tokenize`` (punctuation gets
    glued to neighbours, hyphenated words may split), so words are matched by
    character position in the whitespace-free text rather than one-to-one.
    """
    aligned = [(w, s, e, p) for (w, s, e, p) in aligned if _squash(w)]
    if not aligned or not tokens:
        return [None] * len(tokens)

    token_offsets: List[int] = []
    pos = 0
    for tok in tokens:
        token_offsets.append(pos)
        pos += len(_squash(tok))
    token_total = pos

    word_offsets: List[int] = []
    pos = 0
    for word, _, _, _ in aligned:
        word_offsets.append(pos)
        pos += len(_squash(word))
    word_total = pos
    if token_total == 0 or word_total == 0:
        return [None] * len(tokens)

    scale = word_total / token_total
    times: List[Optional[float]] = []
    for offset in token_offsets:
        target = int(offset * scale + 0.5)
        idx = bisect.bisect_right(word_offsets, target) - 1
        idx = max(0, min(idx, len(aligned) - 1))
        word, start, end, prob = aligned[idx]
        failed = (end - start) <= 0 and prob <= 0
        times.append(None if failed else start)
    return times


def alignment_is_poor(times: Sequence[Optional[float]]) -> bool:
    """True when fewer than half the words got a usable time from the aligner."""
    n = len(times)
    known = sum(1 for t in times if t is not None)
    return known < max(1, (n + 1) // 2)


def fallback_word_times(n_tokens: int, line_start: float, line_end: Optional[float]) -> List[float]:
    """Spread words evenly when the aligner could not place them."""
    if n_tokens == 0:
        return []
    duration = FALLBACK_SECONDS_PER_WORD * n_tokens
    if line_end is not None:
        duration = min(duration, max(line_end - line_start - MIN_WORD_STEP, 0.0))
    return [line_start + duration * i / n_tokens for i in range(n_tokens)]


def finalize_word_times(
    times: Sequence[Optional[float]],
    line_start: float,
    line_end: Optional[float],
) -> Tuple[List[float], bool]:
    """Fill gaps, pin the first word to the line tag, clamp to the window and
    make the sequence strictly increasing.  Returns (times, used_fallback)."""
    n = len(times)
    if n == 0:
        return [], False

    # Too little usable information: the aligner did not really place this line.
    if alignment_is_poor(times):
        return fallback_word_times(n, line_start, line_end), True

    out: List[float] = [0.0] * n
    # Interpolate missing entries between their neighbours.
    filled: List[Optional[float]] = list(times)
    filled[0] = line_start
    i = 1
    while i < n:
        if filled[i] is None:
            j = i
            while j < n and filled[j] is None:
                j += 1
            prev = filled[i - 1]
            nxt = filled[j] if j < n else None
            if nxt is None:
                for k in range(i, n):
                    filled[k] = prev + MIN_WORD_STEP * (k - i + 1) + FALLBACK_SECONDS_PER_WORD * (k - i + 1)
            else:
                span = j - i + 1
                for k in range(i, j):
                    filled[k] = prev + (nxt - prev) * (k - i + 1) / span
            i = j
        else:
            i += 1

    # Clamp into the window, make the sequence non-decreasing, then spread any
    # words that ended up on top of each other evenly between their neighbours.
    upper = None if line_end is None else max(line_end - MIN_WORD_STEP, line_start)
    clamped = [max(float(t), line_start) if upper is None else min(max(float(t), line_start), upper)
               for t in filled]
    clamped[0] = line_start
    out = spread_ties(pool_adjacent_violators(clamped), line_start, line_end)
    for k in range(1, n):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + MIN_WORD_STEP
    return out, False


def pool_adjacent_violators(values: Sequence[float]) -> List[float]:
    """Smallest change that makes ``values`` non-decreasing (isotonic regression)."""
    blocks: List[List[float]] = []  # [sum, count]
    for v in values:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out: List[float] = []
    for s, c in blocks:
        out.extend([s / c] * c)
    return out


def spread_ties(values: Sequence[float], line_start: float, line_end: Optional[float]) -> List[float]:
    """Words sharing one timestamp get spaced evenly up to the next distinct time.

    A trailing group (words pushed against the end of the window) is spaced
    backwards from the previous word instead so the words stay in the gap.
    """
    n = len(values)
    out = list(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[j + 1] - values[i]) < 1e-9:
            j += 1
        size = j - i + 1
        if size > 1:
            v = values[i]
            if j + 1 < n:
                nxt = values[j + 1]
                for k in range(i, j + 1):
                    out[k] = v + (nxt - v) * (k - i) / size
            elif i > 0:
                prev = out[i - 1]
                for k in range(i, j + 1):
                    out[k] = prev + (v - prev) * (k - i + 1) / size
            else:
                span = (line_end - v) if line_end is not None else FALLBACK_SECONDS_PER_WORD * size
                for k in range(i, j + 1):
                    out[k] = v + span * (k - i) / size
        i = j + 1
    return out


def build_word_line(tag: str, tokens: Sequence[str], times: Sequence[float], decimals: int = 2) -> str:
    parts = [f"<{format_timestamp(t, decimals)}>{tok}" for tok, t in zip(tokens, times)]
    return f"{tag}{' '.join(parts)}"


# --------------------------------------------------------------------------- #
# Converting one file
# --------------------------------------------------------------------------- #
@dataclass
class ConvertStats:
    lines_total: int = 0
    lines_aligned: int = 0
    lines_fallback: int = 0
    lines_kept: int = 0
    lines_retimed: int = 0
    # Whole-file shift applied because the lyrics were timed to a different
    # edit of the song (0 when not needed).
    global_offset: float = 0.0
    # (original tag, raw aligned first-word time) for lines that had context;
    # kept for diagnostics and tuning.
    first_word_estimates: List[Tuple[float, float]] = field(default_factory=list)
    # Mean aligner word probability per aligned line.
    line_confidences: List[float] = field(default_factory=list)

    @property
    def confidence(self) -> Optional[float]:
        if not self.line_confidences:
            return None
        return sum(self.line_confidences) / len(self.line_confidences)


def decide_retiming(estimates: Sequence[Tuple[float, float]]) -> Tuple[str, float]:
    """Look at every line's aligned first word and decide what to do.

    Returns ("global", offset) when the whole file is consistently offset from
    the audio, ("per_line", median) when the file runs early but unevenly, and
    ("none", 0.0) when the tags already match the vocals.
    """
    shifts = sorted(e - FIRST_WORD_BIAS - t for t, e in estimates)
    if not shifts:
        return "none", 0.0
    n = len(shifts)
    median = shifts[n // 2] if n % 2 else (shifts[n // 2 - 1] + shifts[n // 2]) / 2
    if n >= GLOBAL_OFFSET_MIN_LINES:
        q1, q3 = shifts[n // 4], shifts[(3 * n) // 4]
        if q3 - q1 <= GLOBAL_OFFSET_MAX_SPREAD and RETIME_DEADBAND <= abs(median) <= MAX_GLOBAL_OFFSET:
            return "global", median
    if median >= RETIME_GATE:
        return "per_line", median
    return "none", 0.0


def shift_line(line: LrcLine, offset: float, decimals: int) -> LrcLine:
    """Move a timed line (and any word tags inside it) by ``offset`` seconds."""
    if line.start is None:
        return line
    new_start = max(0.0, line.start + offset)
    new_tag = f"[{format_timestamp(new_start, decimals)}]"
    text = WORD_TS_RE.sub(
        lambda m: f"<{format_timestamp(max(0.0, parse_timestamp(m.group(1), m.group(2), m.group(3)) + offset), decimals)}>",
        line.text)
    raw = new_tag + ("" if line.has_word_tags or not text else " ") + text
    return LrcLine(raw=raw, start=new_start, tag=new_tag, text=text, has_word_tags=line.has_word_tags)


def next_line_start(lines: Sequence[LrcLine], start: float) -> Optional[float]:
    """Timestamp of the first timed line after ``start`` (any kind of line)."""
    later = [ln.start for ln in lines if ln.start is not None and ln.start > start]
    return min(later) if later else None


def make_room_for_tails(lines: Sequence[LrcLine], raw_times: Dict[int, List[Optional[float]]],
                        eff_start: Dict[int, float]) -> int:
    """Push an early tag later when the previous line's last word needs the room.

    Uses two lines' evidence at once: line A's aligned last word and line B's
    aligned first word.  Returns how many tags moved.  ``eff_start`` is updated
    in place.
    """
    moved = 0
    order = sorted(raw_times, key=lambda i: lines[i].start)
    for a, b in zip(order, order[1:]):
        ra, rb = raw_times[a], raw_times[b]
        if alignment_is_poor(ra) or alignment_is_poor(rb):
            continue
        last_a = next((t for t in reversed(ra) if t is not None), None)
        first_b = next((t for t in rb if t is not None), None)
        if last_a is None or first_b is None:
            continue
        tag_b = eff_start[b]
        first_b -= FIRST_WORD_BIAS
        if last_a < tag_b - TAIL_ROOM_TRIGGER or first_b <= last_a + TAIL_ROOM:
            continue
        new = min(last_a + TAIL_ROOM, first_b, float(lines[b].start) + MAX_TAIL_SHIFT)
        after = next_line_start(lines, float(lines[b].start))
        if after is not None:
            new = min(new, after - MIN_WORD_STEP * 2)
        if new > tag_b + 0.005:
            eff_start[b] = new
            moved += 1
    return moved


def retimed_line_start(first_word: Optional[float], tag_start: float, next_start: Optional[float]) -> float:
    """Move a line tag to where its first word is sung, within safe bounds.

    Lyric files are often tagged a few tenths of a second before the vocal
    actually starts, so the player flips to the line too early.  The shift is
    bounded so a bad alignment cannot drag a line far from its original place,
    and the tag never reaches the next line's tag.
    """
    if first_word is None:
        return tag_start
    if first_word - tag_start < RETIME_DEADBAND:
        return tag_start
    new_start = min(first_word, tag_start + MAX_LATE_SHIFT)
    if next_start is not None:
        new_start = min(new_start, next_start - MAX_EARLY_SHIFT - MIN_WORD_STEP)
    return max(new_start, tag_start - MAX_EARLY_SHIFT, 0.0)


def line_windows(lines: Sequence[LrcLine], audio_duration: Optional[float]) -> Dict[int, Tuple[float, float]]:
    """Return {line index: (window_start, window_end)} for lines needing alignment.

    A line's window runs from its own timestamp to the next timed line (of any
    kind, so an instrumental marker closes a window), capped at 30 seconds and
    at the end of the audio.
    """
    timed = sorted((ln.start, idx) for idx, ln in enumerate(lines) if ln.start is not None)
    starts = [s for s, _ in timed]
    windows: Dict[int, Tuple[float, float]] = {}
    for pos, (start, idx) in enumerate(timed):
        if not lines[idx].needs_alignment:
            continue
        # Skip over lines that share the exact same timestamp.
        nxt = bisect.bisect_right(starts, start)
        end = starts[nxt] if nxt < len(starts) else start + MAX_LINE_WINDOW
        end = min(end, start + MAX_LINE_WINDOW)
        if audio_duration is not None:
            end = min(end, audio_duration)
        if end <= start + MIN_WORD_STEP:
            end = start + MIN_WORD_STEP * 2
        windows[idx] = (start, end)
    return windows


@dataclass
class Context:
    slice_start: float
    slice_end: float
    text: str
    tokens: List[str]
    first_token: int      # index in ``tokens`` where the line being aligned starts
    has_prev: bool


def clean_text(line: LrcLine) -> str:
    return WORD_TS_RE.sub("", line.text).strip()


def build_context(lines: Sequence[LrcLine], idx: int, window_end: float,
                  audio_duration: Optional[float]) -> Context:
    """Choose the audio slice and text used to align line ``idx``.

    Whisper pulls the first word of a slice to the slice start and the last
    word towards its end, so the neighbouring lyric lines are aligned in the
    same call whenever they are close: the line then sits in the middle of the
    slice with real words on both sides of it.
    """
    line = lines[idx]
    start = float(line.start)
    lyric = sorted((ln.start, i) for i, ln in enumerate(lines) if ln.is_lyric)
    pos = next(k for k, (_, i) in enumerate(lyric) if i == idx)

    prev_i = lyric[pos - 1][1] if pos > 0 else None
    if prev_i is not None and start - lines[prev_i].start > CONTEXT_GAP:
        prev_i = None
    next_i = lyric[pos + 1][1] if pos + 1 < len(lyric) else None
    if next_i is not None and lines[next_i].start - start > CONTEXT_GAP:
        next_i = None
    after_i = lyric[pos + 2][1] if next_i is not None and pos + 2 < len(lyric) else None

    def slice_bounds(with_prev: bool, with_next: bool) -> Tuple[float, float]:
        s = lines[prev_i].start - LEAD_PAD if with_prev else start - LEAD_PAD_SOLO
        if with_next:
            ns = lines[next_i].start
            e = ns + CONTEXT_TAIL
            if after_i is not None:
                e = min(e, lines[after_i].start)
            e = max(e, ns + TAIL_PAD)
        else:
            # No next line close by: run up to the next tag (a slow ballad line
            # can take ten seconds), but never more than SOLO_TAIL_MAX.
            e = min(window_end + TAIL_PAD, start + SOLO_TAIL_MAX)
        return max(0.0, s), e

    s, e = slice_bounds(prev_i is not None, next_i is not None)
    if e - s > MAX_LINE_WINDOW and next_i is not None:
        next_i = None
        s, e = slice_bounds(prev_i is not None, False)
    if e - s > MAX_LINE_WINDOW and prev_i is not None:
        prev_i = None
        s, e = slice_bounds(False, False)
    e = min(e, s + MAX_LINE_WINDOW)
    if audio_duration is not None:
        e = min(e, max(audio_duration, s + MIN_WORD_STEP * 2))

    parts: List[str] = []
    prev_tokens = 0
    if prev_i is not None:
        parts.append(clean_text(lines[prev_i]))
        prev_tokens = len(tokenize(parts[-1]))
    parts.append(line.text)
    if next_i is not None:
        parts.append(clean_text(lines[next_i]))
    tokens = [t for p in parts for t in tokenize(p)]
    return Context(s, e, "\n".join(parts), tokens, prev_tokens, prev_i is not None)


def convert_lines(
    lines: Sequence[LrcLine],
    align_fn: AlignFn,
    audio_duration: Optional[float] = None,
    decimals: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
    retime_lines: bool = True,
    _allow_global_shift: bool = True,
) -> Tuple[List[str], ConvertStats]:
    stats = ConvertStats(lines_total=len(lines))
    windows = line_windows(lines, audio_duration)

    # Pass 1: align every line and decide where its tag should sit.
    raw_times: Dict[int, List[Optional[float]]] = {}
    tokens_by_idx: Dict[int, List[str]] = {}
    first_words: Dict[int, float] = {}
    eff_start: Dict[int, float] = {idx: ln.start for idx, ln in enumerate(lines) if ln.start is not None}
    for idx in windows:
        if should_stop and should_stop():
            raise InterruptedError("stopped")
        start, end = windows[idx]
        ctx = build_context(lines, idx, end, audio_duration)
        tokens = tokenize(lines[idx].text)
        try:
            aligned = align_fn(ctx.slice_start, ctx.slice_end, ctx.text)
        except Exception:  # noqa: BLE001 - one bad line must not kill the file
            aligned = []
        all_times = map_aligned_words_to_tokens(ctx.tokens, aligned)
        times = all_times[ctx.first_token:ctx.first_token + len(tokens)]
        raw_times[idx] = times
        tokens_by_idx[idx] = tokens
        probs = [float(p) for (_, _, _, p) in aligned if p is not None]
        if probs:
            stats.line_confidences.append(sum(probs) / len(probs))
        # Only trust the first word's time when a previous line was aligned in
        # front of it: the first word of a slice is always pulled to the slice start.
        if ctx.has_prev and times and times[0] is not None and not alignment_is_poor(times):
            stats.first_word_estimates.append((start, times[0]))
            first_words[idx] = times[0]

    if retime_lines:
        mode, amount = decide_retiming(stats.first_word_estimates)
        if mode == "global" and _allow_global_shift:
            # The whole file is offset: shift every tag and align again so the
            # audio slices are taken from the right places.
            shifted = [shift_line(ln, amount, decimals) for ln in lines]
            output, stats2 = convert_lines(shifted, align_fn, audio_duration, decimals, should_stop,
                                           retime_lines=True, _allow_global_shift=False)
            stats2.global_offset = amount
            return output, stats2
        if mode in ("global", "per_line"):
            for idx, first in first_words.items():
                start = float(lines[idx].start)
                new_start = retimed_line_start(first - FIRST_WORD_BIAS, start, next_line_start(lines, start))
                if abs(new_start - start) >= 0.005:
                    stats.lines_retimed += 1
                eff_start[idx] = new_start
        stats.lines_retimed += make_room_for_tails(lines, raw_times, eff_start)

    # Pass 2: every word must sit between its own (possibly moved) tag and the
    # next line's tag, otherwise the player flips lines before showing it.
    ordered = sorted(eff_start.values())
    pace = song_pace(raw_times.values())
    output: List[str] = []
    for idx, line in enumerate(lines):
        if idx not in windows:
            output.append(line.raw)
            stats.lines_kept += 1
            continue
        start = eff_start[idx]
        pos = bisect.bisect_right(ordered, start)
        end = ordered[pos] if pos < len(ordered) else start + MAX_LINE_WINDOW
        end = min(end, start + MAX_LINE_WINDOW)
        if audio_duration is not None:
            end = min(end, audio_duration)
        if end <= start + MIN_WORD_STEP:
            end = start + MIN_WORD_STEP * 2
        final_times, used_fallback = finalize_word_times(raw_times[idx], start, end)
        if used_fallback:
            stats.lines_fallback += 1
        else:
            stats.lines_aligned += 1
            final_times = cap_line_span(final_times, pace=pace)
        tag = line.tag if abs(start - line.start) < 0.005 else f"[{format_timestamp(start, decimals)}]"
        output.append(build_word_line(tag, tokens_by_idx[idx], final_times, decimals))
    return output, stats


def detect_decimals(lines: Sequence[LrcLine]) -> int:
    """Use 3 decimals when the file already uses millisecond tags, else 2."""
    for ln in lines:
        if ln.tag:
            m = LINE_TS_RE.match(ln.tag)
            if m and m.group(3) and len(m.group(3)) == 3:
                return 3
    return 2


# --------------------------------------------------------------------------- #
# Finding files
# --------------------------------------------------------------------------- #
def normalize_stem(name: str) -> str:
    """Normalise a file stem for matching lyrics to audio.

    Handles ``#U2019``-style escapes that some zip tools write for non-ASCII
    characters, Unicode normalisation forms, case and whitespace differences.
    """
    name = ZIP_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), name)
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"\s+", " ", name).strip().casefold()
    return name


def iter_files(folder: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in sorted(files):
                yield Path(root) / f
    else:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                yield f


def index_audio(folder: Path, recursive: bool) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for f in iter_files(folder, recursive):
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(normalize_stem(f.stem), f)
    return index


def find_lrc_files(folder: Path, recursive: bool) -> List[Path]:
    return [f for f in iter_files(folder, recursive) if f.suffix.lower() == ".lrc"]


def match_audio(lrc_path: Path, audio_index: Dict[str, Path], lyrics_dir: Path, audio_dir: Path) -> Optional[Path]:
    """Prefer an audio file sitting next to the LRC, then fall back to the index."""
    for candidate in lrc_path.parent.iterdir() if lrc_path.parent.exists() else []:
        if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.stem == lrc_path.stem:
            return candidate
    key = normalize_stem(lrc_path.stem)
    if key in audio_index:
        return audio_index[key]
    # Some rippers cut long names; try prefix matching as a last resort.
    if len(key) >= 12:
        prefixed = [p for k, p in audio_index.items() if k.startswith(key) or key.startswith(k)]
        if len(prefixed) == 1:
            return prefixed[0]
    return None


# --------------------------------------------------------------------------- #
# Audio loading (ffmpeg on PATH, or the one bundled with imageio-ffmpeg)
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE):
    """Decode any audio file to a mono float32 numpy array at 16 kHz."""
    import numpy as np

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and add it to PATH, or run "
            "'pip install imageio-ffmpeg' to use a bundled copy."
        )
    cmd = [
        ffmpeg, "-nostdin", "-threads", "0", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to decode audio: {e.stderr.decode(errors='ignore').strip()[-500:]}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# The whisper based aligner
# --------------------------------------------------------------------------- #
class WhisperLineAligner:
    """Force-aligns lyric text to audio using stable-ts on top of Whisper."""

    def __init__(self, model_name: str = "base", device: Optional[str] = None,
                 download_root: Optional[str] = None):
        import stable_whisper
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = stable_whisper.load_model(model_name, device=self.device, download_root=download_root)
        self._apply_alignment_heads(model_name)
        self.audio = None
        self.audio_duration = 0.0

    def _apply_alignment_heads(self, model_name: str) -> None:
        """Models loaded from a file path miss whisper's tuned alignment heads.

        Whisper only applies them when loading by name, so look them up from
        the file stem (``base.en.pt`` -> ``base.en``) and apply them ourselves.
        """
        path = Path(model_name)
        if not path.suffix.lower() == ".pt":
            return
        try:
            from whisper import _ALIGNMENT_HEADS

            heads = _ALIGNMENT_HEADS.get(path.stem)
            if heads is not None:
                self.model.set_alignment_heads(heads)
        except Exception:  # noqa: BLE001 - keep working with the default heads
            pass

    def set_audio(self, audio) -> None:
        self.audio = audio
        self.audio_duration = len(audio) / SAMPLE_RATE

    def load_file(self, path: Path) -> float:
        self.set_audio(load_audio(path))
        return self.audio_duration

    def detect_language(self, start: float = 0.0) -> str:
        import whisper

        a, b = int(start * SAMPLE_RATE), int((start + 30.0) * SAMPLE_RATE)
        clip = whisper.pad_or_trim(self.audio[a:b])
        mel = whisper.log_mel_spectrogram(clip, n_mels=self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def align(self, start: float, end: float, text: str, language: str = "en"
              ) -> List[Tuple[str, float, float, float]]:
        if self.audio is None:
            raise RuntimeError("No audio loaded")
        a = max(0, int(start * SAMPLE_RATE))
        b = min(len(self.audio), int(end * SAMPLE_RATE))
        clip = self.audio[a:b]
        if len(clip) < SAMPLE_RATE // 10:
            return []
        # stable-ts always shows an "Adjustment" progress bar; keep the console clean.
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            result = self.model.align(
                clip, text, language=language, original_split=True,
                verbose=None, regroup=False, ignore_compatibility=True,
            )
        if result is None:
            return []
        offset = a / SAMPLE_RATE
        words: List[Tuple[str, float, float, float]] = []
        for w in result.all_words():
            words.append((w.word, w.start + offset, w.end + offset, float(getattr(w, "probability", 1.0) or 0.0)))
        return words

    def make_align_fn(self, language: str) -> AlignFn:
        return lambda start, end, text: self.align(start, end, text, language)


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
@dataclass
class FileResult:
    lrc: Path
    status: str                     # converted | skipped_word_level | skipped_no_audio | skipped_no_lyrics | failed
    output: Optional[Path] = None
    audio: Optional[Path] = None
    stats: Optional[ConvertStats] = None
    message: str = ""

    def flags(self) -> List[str]:
        """Reasons a person should look at this file."""
        out: List[str] = []
        if self.status == "failed":
            out.append(f"failed: {self.message}")
        elif self.status == "skipped_no_audio":
            out.append("no matching audio file")
        elif self.status == "converted" and self.stats:
            s = self.stats
            if abs(s.global_offset) >= 0.005:
                out.append(f"lyrics shifted {s.global_offset:+.2f} s to match the audio "
                           "(lyric file timed to a different edit of the song?)")
            aligned = s.lines_aligned + s.lines_fallback
            if s.lines_fallback and aligned and s.lines_fallback / aligned >= 0.2:
                out.append(f"{s.lines_fallback} of {aligned} lines could not be aligned and were spaced evenly")
            elif s.lines_fallback:
                out.append(f"{s.lines_fallback} line(s) could not be aligned and were spaced evenly")
            if s.confidence is not None and s.confidence < LOW_CONFIDENCE:
                out.append(f"low alignment confidence ({s.confidence:.2f}); word timing may be rough "
                           "(loud or unclear vocals?) - try a larger model")
        return out


@dataclass
class Summary:
    results: List[FileResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    def describe(self) -> str:
        return (
            f"converted {self.count('converted')}, "
            f"already word-by-word {self.count('skipped_word_level')}, "
            f"no matching audio {self.count('skipped_no_audio')}, "
            f"no lyrics {self.count('skipped_no_lyrics')}, "
            f"failed {self.count('failed')}"
        )

    def flagged(self) -> List[FileResult]:
        return [r for r in self.results if r.flags()]


REPORT_NAME = "lrc_conversion_report.txt"
# A run limited to chosen files writes here so the full report survives.
RERUN_REPORT_NAME = "lrc_conversion_report_rerun.txt"


def flagged_names_from_report(report_path: Path) -> List[str]:
    """File names listed in the FLAGGED section of a report written by this tool."""
    names: List[str] = []
    in_flagged = False
    for line in read_text(report_path).splitlines():
        if line.startswith("FLAGGED"):
            in_flagged = True
            continue
        if line.startswith("ALL FILES"):
            break
        if in_flagged and line.startswith("  ") and not line.startswith("      -") and line.strip() != "none":
            names.append(line.strip())
    return names


def write_report(summary: Summary, output_dir: Path, name: str = REPORT_NAME) -> Path:
    """Write a plain-text report with the files worth checking listed first."""
    lines = ["LRC line-to-word conversion report", summary.describe(), ""]
    flagged = summary.flagged()
    lines.append(f"FLAGGED ({len(flagged)}) - worth checking by hand:")
    if not flagged:
        lines.append("  none")
    for r in flagged:
        lines.append(f"  {r.lrc.name}")
        for f in r.flags():
            lines.append(f"      - {f}")
    lines += ["", "ALL FILES:"]
    for r in summary.results:
        detail = ""
        if r.stats:
            s = r.stats
            detail = f"  aligned={s.lines_aligned} fallback={s.lines_fallback} retimed={s.lines_retimed}"
            if s.confidence is not None:
                detail += f" confidence={s.confidence:.2f}"
            if abs(s.global_offset) >= 0.005:
                detail += f" offset={s.global_offset:+.2f}s"
        audio = f"  audio={r.audio.name}" if r.audio else ""
        lines.append(f"  {r.status:20s} {r.lrc.name}{audio}{detail}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def output_path_for(lrc: Path, lyrics_dir: Path, output_dir: Path, recursive: bool) -> Path:
    if recursive:
        try:
            return output_dir / lrc.relative_to(lyrics_dir)
        except ValueError:
            pass
    return output_dir / lrc.name


def written_by_this_tool(lines: Sequence[LrcLine]) -> bool:
    """True when the file's first ``[re:...]`` tag names this tool as the writer."""
    for ln in lines:
        m = PROVENANCE_RE.match(ln.raw)
        if m:
            payload = m.group(1).split()
            return bool(payload) and payload[0] == WRITER
    return False


def line_level_source(content: str) -> str:
    """Turn one of this tool's word-by-word files back into line-by-line text."""
    out = []
    for line in content.splitlines():
        if PROVENANCE_RE.match(line):
            continue
        m = LINE_TS_RE.match(line.strip())
        if m and WORD_TS_RE.search(line):
            text = WORD_TS_RE.sub("", line.strip()[m.end():]).strip()
            out.append(f"{m.group(0)} {text}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def stamp_provenance(out_lines: Sequence[str]) -> List[str]:
    """Put this tool's ``[re:...]`` tag at the top of a converted file.

    Any existing ``[re:...]`` line is removed rather than copied through: it
    would otherwise still name whoever wrote the line-level file and claim the
    timing is line-level, which is no longer true.  Where that old tag named a
    source for the words it is kept as a fifth field (``from-<source>``),
    which readers ignore.  The timing field describes the file actually
    written, so a file that somehow ends up without word tags says ``line``.
    """
    old_payload: Optional[List[str]] = None
    kept: List[str] = []
    for line in out_lines:
        m = PROVENANCE_RE.match(line)
        if m:
            if old_payload is None:
                old_payload = m.group(1).split()
            continue
        kept.append(line)
    timing = "word" if any(WORD_TS_RE.search(line) for line in kept) else "line"
    fields = [WRITER, __version__, "align", timing]
    if old_payload and len(old_payload) >= 3 and old_payload[2] and old_payload[2] != "align":
        fields.append(f"from-{old_payload[2]}")
    return [f"[re:{' '.join(fields)}]"] + kept


def backup_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst``, contents only.

    ``shutil.copy2`` also copies permissions and timestamps, which network
    mounts such as gvfs SMB shares reject with "Operation not supported".
    """
    try:
        shutil.copy2(src, dst)
    except OSError:
        if dst.exists():
            dst.unlink()
        shutil.copyfile(src, dst)


def convert_file(
    lrc_path: Path,
    audio_path: Path,
    aligner: WhisperLineAligner,
    output_path: Path,
    language: str = "en",
    should_stop: Optional[Callable[[], bool]] = None,
    retime_lines: bool = True,
    content: Optional[str] = None,
) -> FileResult:
    if content is None:
        content = read_text(lrc_path)
    lines = parse_lrc(content)
    if not any(ln.is_lyric for ln in lines):
        return FileResult(lrc_path, "skipped_no_lyrics", message="no timed lyric lines")
    if is_word_level(lines):
        return FileResult(lrc_path, "skipped_word_level")

    duration = aligner.load_file(audio_path)
    lang = language
    if not lang or lang == "auto":
        first = min((ln.start for ln in lines if ln.needs_alignment), default=0.0)
        lang = aligner.detect_language(first)

    new_lines, stats = convert_lines(
        lines, aligner.make_align_fn(lang), audio_duration=duration,
        decimals=detect_decimals(lines), should_stop=should_stop, retime_lines=retime_lines,
    )

    new_lines = stamp_provenance(new_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == lrc_path.resolve():
        backup = lrc_path.with_suffix(lrc_path.suffix + ".bak")
        if not backup.exists():
            backup_file(lrc_path, backup)
    newline = "\r\n" if "\r\n" in content else "\n"
    output_path.write_text(newline.join(new_lines) + newline, encoding="utf-8")
    return FileResult(lrc_path, "converted", output=output_path, audio=audio_path, stats=stats,
                      message=f"language={lang}")


def process_library(
    audio_dir: Path,
    lyrics_dir: Path,
    output_dir: Path,
    aligner: WhisperLineAligner,
    language: str = "en",
    recursive: bool = False,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    retime_lines: bool = True,
    reconvert: bool = False,
    only: Optional[Iterable[str]] = None,
) -> Summary:
    """Convert every line-level .lrc under ``lyrics_dir``.

    With ``reconvert`` the files this tool converted earlier (recognised by
    their ``[re:lrc-align ...]`` tag) are done again from the ``.lrc.bak``
    original where one exists, otherwise from the file with its word tags
    stripped.  Word-by-word files written by anything else are never touched.
    ``only`` limits the run to lyric files with those names (for example the
    flagged names from an earlier report); other files are not even listed.
    """
    summary = Summary()
    lrc_files = find_lrc_files(lyrics_dir, recursive)
    if only is not None:
        wanted = {n.strip() for n in only}
        lrc_files = [f for f in lrc_files if f.name in wanted]
        missing = wanted - {f.name for f in lrc_files}
        for name in sorted(missing):
            log(f"Not found in the lyrics folder: {name}")
    if not lrc_files:
        log("No .lrc files found in the lyrics folder.")
        return summary
    audio_index = index_audio(audio_dir, recursive)
    log(f"Found {len(lrc_files)} lyric files and {len(audio_index)} audio files.")

    total = len(lrc_files)
    for i, lrc in enumerate(lrc_files, 1):
        if should_stop and should_stop():
            log("Stopped.")
            break
        out_path = output_path_for(lrc, lyrics_dir, output_dir, recursive)
        try:
            if out_path.resolve() != lrc.resolve() and out_path.exists():
                existing = parse_lrc(read_text(out_path))
                if is_word_level(existing) and not (reconvert and written_by_this_tool(existing)):
                    result = FileResult(lrc, "skipped_word_level", output=out_path,
                                        message="output already word-by-word")
                    summary.results.append(result)
                    log(f"[{i}/{total}] Skip (output already word-by-word): {lrc.name}")
                    if progress:
                        progress(i, total)
                    continue

            content = read_text(lrc)
            lines = parse_lrc(content)
            redo = False
            if reconvert and is_word_level(lines) and written_by_this_tool(lines):
                backup = lrc.with_suffix(lrc.suffix + ".bak")
                content = read_text(backup) if backup.exists() else line_level_source(content)
                lines = parse_lrc(content)
                redo = True
            if not any(ln.is_lyric for ln in lines):
                result = FileResult(lrc, "skipped_no_lyrics")
                log(f"[{i}/{total}] Skip (no timed lyrics): {lrc.name}")
            elif is_word_level(lines):
                result = FileResult(lrc, "skipped_word_level")
                log(f"[{i}/{total}] Skip (already word-by-word): {lrc.name}")
            else:
                audio = match_audio(lrc, audio_index, lyrics_dir, audio_dir)
                if audio is None:
                    result = FileResult(lrc, "skipped_no_audio")
                    log(f"[{i}/{total}] Skip (no matching audio): {lrc.name}")
                else:
                    log(f"[{i}/{total}] {'Re-converting' if redo else 'Aligning'}: {lrc.name}  <-  {audio.name}")
                    result = convert_file(lrc, audio, aligner, out_path, language, should_stop, retime_lines,
                                          content=content)
                    s = result.stats
                    if s:
                        extra = f", {s.lines_retimed} line time(s) adjusted" if s.lines_retimed else ""
                        if abs(s.global_offset) >= 0.005:
                            extra += f", whole file shifted {s.global_offset:+.2f} s"
                        if s.lines_fallback:
                            extra += f", {s.lines_fallback} line(s) fell back to even spacing"
                        log(f"    Saved {out_path.name}: {s.lines_aligned} line(s) aligned{extra}")
        except InterruptedError:
            log("Stopped.")
            break
        except Exception as e:  # noqa: BLE001
            result = FileResult(lrc, "failed", message=str(e))
            log(f"[{i}/{total}] FAILED {lrc.name}: {e}")
        summary.results.append(result)
        if progress:
            progress(i, total)
    try:
        report = write_report(summary, output_dir, REPORT_NAME if only is None else RERUN_REPORT_NAME)
        log(f"Report written to {report}")
        if summary.flagged():
            log(f"{len(summary.flagged())} file(s) flagged for checking, see the report.")
    except OSError as e:
        log(f"Could not write report: {e}")
    return summary
