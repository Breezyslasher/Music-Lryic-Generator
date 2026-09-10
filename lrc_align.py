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

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".aiff", ".aif", ".alac", ".wv", ".ape", ".mp4",
}

SAMPLE_RATE = 16000

# Longest audio window used for a single line.  Whisper works on 30 second
# chunks, and a lyric line never needs more than that.
MAX_LINE_WINDOW = 30.0
# Audio included before the line timestamp so a clipped first consonant is not lost.
LEAD_PAD = 0.15
# Minimum spacing enforced between consecutive word tags.
MIN_WORD_STEP = 0.01
# Rough spoken/sung duration per word used when alignment fails for a line.
FALLBACK_SECONDS_PER_WORD = 0.45

LINE_TS_RE = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")
WORD_TS_RE = re.compile(r"<(\d+):(\d{1,2})(?:[.:](\d{1,3}))?>")
# ID tags such as [ar:Artist], [ti:Title], [offset:+200]
META_RE = re.compile(r"^\s*\[([A-Za-z#][^\]:]*):([^\]]*)\]\s*$")
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

    known = [t for t in times if t is not None]
    # Too little usable information: the aligner did not really place this line.
    if len(known) < max(1, (n + 1) // 2):
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

    prev = line_start
    out[0] = line_start
    for k in range(1, n):
        t = float(filled[k])
        t = max(t, prev + MIN_WORD_STEP)
        if line_end is not None:
            # Leave room for the words that still follow this one.
            upper = line_end - MIN_WORD_STEP * (n - k)
            if t > upper:
                t = max(upper, prev + MIN_WORD_STEP)
        out[k] = t
        prev = t
    return out, False


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


def convert_lines(
    lines: Sequence[LrcLine],
    align_fn: AlignFn,
    audio_duration: Optional[float] = None,
    decimals: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[str], ConvertStats]:
    stats = ConvertStats(lines_total=len(lines))
    windows = line_windows(lines, audio_duration)
    output: List[str] = []
    for idx, line in enumerate(lines):
        if idx not in windows:
            output.append(line.raw)
            stats.lines_kept += 1
            continue
        if should_stop and should_stop():
            raise InterruptedError("stopped")
        start, end = windows[idx]
        tokens = tokenize(line.text)
        try:
            aligned = align_fn(max(0.0, start - LEAD_PAD), end, line.text)
        except Exception:  # noqa: BLE001 - one bad line must not kill the file
            aligned = []
        times = map_aligned_words_to_tokens(tokens, aligned)
        final_times, used_fallback = finalize_word_times(times, start, end)
        if used_fallback:
            stats.lines_fallback += 1
        else:
            stats.lines_aligned += 1
        output.append(build_word_line(line.tag, tokens, final_times, decimals))
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
        self.audio = None
        self.audio_duration = 0.0

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


def output_path_for(lrc: Path, lyrics_dir: Path, output_dir: Path, recursive: bool) -> Path:
    if recursive:
        try:
            return output_dir / lrc.relative_to(lyrics_dir)
        except ValueError:
            pass
    return output_dir / lrc.name


def convert_file(
    lrc_path: Path,
    audio_path: Path,
    aligner: WhisperLineAligner,
    output_path: Path,
    language: str = "en",
    should_stop: Optional[Callable[[], bool]] = None,
) -> FileResult:
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
        decimals=detect_decimals(lines), should_stop=should_stop,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == lrc_path.resolve():
        backup = lrc_path.with_suffix(lrc_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(lrc_path, backup)
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
) -> Summary:
    summary = Summary()
    lrc_files = find_lrc_files(lyrics_dir, recursive)
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
                if is_word_level(existing):
                    result = FileResult(lrc, "skipped_word_level", output=out_path,
                                        message="output already word-by-word")
                    summary.results.append(result)
                    log(f"[{i}/{total}] Skip (output already word-by-word): {lrc.name}")
                    if progress:
                        progress(i, total)
                    continue

            lines = parse_lrc(read_text(lrc))
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
                    log(f"[{i}/{total}] Aligning: {lrc.name}  <-  {audio.name}")
                    result = convert_file(lrc, audio, aligner, out_path, language, should_stop)
                    s = result.stats
                    if s:
                        extra = f", {s.lines_fallback} line(s) fell back to even spacing" if s.lines_fallback else ""
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
    return summary
