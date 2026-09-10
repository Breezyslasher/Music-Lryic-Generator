# CLAUDE.md — Music-Lryic-Generator

## What this project is

A forced aligner. It takes a line-level `.lrc` and the matching audio, and
writes a word-by-word (Enhanced / A2) `.lrc` by aligning the *existing* text
against the audio. It does not transcribe: the words are already correct, only
the timing is being established.

## The task: stamp the output with a provenance tag

The converted files are consumed by Beetdrop
(github.com/Breezyslasher/beetdrop), which files tagged audio into a music
library and writes `.lrc` sidecars of its own. Beetdrop records who wrote each
sidecar, and reasons about the library from it — which files a repair pass
should redo, which sources have coverage, which files to leave alone. Right now
this tool's output is indistinguishable from Beetdrop's own, which makes that
reasoning wrong in two specific ways described below.

### Fix first: the existing tag is copied through, and becomes a lie

`convert_lines` in `lrc_align.py` copies every non-lyric line through
verbatim:

```python
for idx, line in enumerate(lines):
    if idx not in windows:
        output.append(line.raw)      # <- metadata lines, unchanged
```

Beetdrop writes an id tag at the top of every sidecar it creates:

```
[re:beetdrop 0.62.0 apple line]
[00:28.90]Tumble out of bed
```

So converting one of its files today produces:

```
[re:beetdrop 0.62.0 apple line]                        <- still says line-level
[00:28.90]<00:28.90>Tumble <00:29.34>out ...           <- but it is word-level now
```

The file now claims Beetdrop rendered it, that the timing came from Apple, and
that it is line-level. All three are false. Beetdrop trusts that tag, so a pass
looking for "files an old build rendered badly" will skip this one, and Stats
will credit Apple for word timing Apple never supplied.

**Replace any existing `[re:...]` line rather than copying it through.** Never
emit two — Beetdrop reads the first match and ignores the rest.

### The format

Beetdrop parses it with, effectively:

```python
_SOURCE_TAG = re.compile(r"^\[re:([^\]]*)\]\r?\n?", re.M)
# payload.split() -> [writer, version, source, timing]
```

`[re:]` is the standard LRC id tag for "the program that created this file", so
players already ignore it. Requirements, all verified against Beetdrop's parser:

- One line, starting at column 0, `[re:` … `]`, no `]` inside.
- Four whitespace-separated fields. Extra fields are tolerated and ignored, so
  a fifth is a safe place to keep something; fewer than four read as empty
  strings, which is fine but less useful.
- Anywhere before the first timestamped line. Put it with the other id tags
  (`[ar:]`, `[ti:]`), at the top.

### What to write

```
[re:lrc-align <version> align word]
```

- **writer** — `lrc-align`. Beetdrop understands any writer's tag, not only its
  own, and a file that names its writer is one Beetdrop will not guess about.
- **version** — this tool's version. There is no `__version__` in the project
  yet; add one (a module-level constant in `lrc_align.py` is enough) rather than
  hardcoding a literal in the tag. Use `?` only if that is genuinely unknown.
- **source** — `align`. The field means "where this file's timing came from",
  and for an aligned file that is this tool, not whatever database supplied the
  words. Writing `apple` or `lrclib` here would re-create exactly the confusion
  above. Beetdrop groups Stats by this field, so `align` gives aligned files
  their own visible bucket.
- **timing** — `word` for a converted file. It must describe the file that was
  actually written, so if a file came out with no word tags at all, do not
  claim `word`.

If you want to keep where the words originally came from, carry it as a fifth
field — `[re:lrc-align 1.2 align word from-lrclib]`. Beetdrop ignores it, and
it is better than losing it when the old tag is replaced.

### Where it goes

- Stamp only files this tool actually converted. A file skipped for already
  being word-by-word must be left byte-identical — the tool's "safe to re-run"
  property depends on that, and re-running should not start rewriting headers.
- Stamp the output whether it goes to `--output DIR` or in place. The `.lrc.bak`
  an in-place run leaves is what Beetdrop currently uses as a fallback signal;
  a real tag makes that guesswork unnecessary, and an `--output` run leaves no
  `.bak` at all.
- The tag belongs in the file-writing path, alongside the existing
  metadata-copying, not in the alignment code.

### Do not

- Change any timing behaviour. Line retiming (≤0.35 s later), the whole-file
  offset shift and the even-spacing fallback are all deliberate and are not
  part of this task.
- Add a second `[re:]` line, or a non-standard tag like `[tool:]` — Beetdrop
  will not read it and some players render unknown brackets as lyric text.
- Put the tag after the first timestamp. It parses, but no other LRC writer
  does that.

### Tests

`tests/test_lrc_align.py` exists; add to it:

- A converted file has exactly one `[re:]` line, and it parses into
  `["lrc-align", <version>, "align", "word"]`.
- An input carrying `[re:beetdrop 0.62.0 apple line]` comes out with that line
  **gone**, replaced by this tool's — not both.
- A file skipped for already being word-by-word is byte-identical afterwards.
- Converting twice is idempotent: the second run changes nothing.
- The lyrics and every timestamp are unchanged by the tagging itself — assert
  the file with its `[re:]` line removed matches what the untagged run produced.
- An input with no `[re:]` line still gets one.

### Checking it against Beetdrop

Beetdrop's reader, if you want to confirm a produced file end to end:

```python
from beetdrop.lyrics import lyric_provenance
lyric_provenance(open("Song.lrc").read())
# Provenance(writer='lrc-align', version='1.2', source='align', timing='word')
```

Beetdrop's `scan-lyrics --tag-sources` pass must then report the file as
already tagged and leave it untouched.
