# LRC Line-to-Word Converter

Turns line-by-line `.lrc` lyric files into word-by-word (enhanced) `.lrc` files
for apps like Plex or Music Assistant.

The tool does **not** transcribe your songs. Whisper transcriptions get words
wrong; your existing lyric files already have the right words. Instead, it takes
each timed line and force-aligns that exact text against the piece of the audio
that belongs to the line (from the line's timestamp up to the next line's
timestamp). Every word then gets its own inline timestamp:

```
[00:28.90] I got a feeling that tonight's gonna be a good night
```

becomes

```
[00:28.90]<00:28.90>I <00:29.34>got <00:29.62>a <00:29.81>feeling <00:32.99>that <00:33.24>tonight's <00:33.95>gonna <00:34.47>be <00:34.93>a <00:35.42>good <00:35.88>night
```

What the tool keeps and skips:

- Files that are already word-by-word are skipped untouched.
- Line timestamps are never changed; the first word always gets the line's time.
- Metadata tags (`[ar:...]`, `[ti:...]`), blank lines and instrumental markers
  (a timestamp with no text) are copied through as they are.
- In a mixed file, lines that already have word tags are kept and only the
  plain lines are converted.
- If the aligner cannot place a line, its words are spread evenly from the line
  timestamp and the log says so.

## Setup

### 1. Install Python

Python 3.8 or newer: https://www.python.org/downloads/

    python --version

### 2. Install ffmpeg

Whisper needs ffmpeg to decode audio. Either install it and add it to your
PATH, or install the bundled copy with pip (included in `requirements.txt`):

    pip install imageio-ffmpeg

### 3. Install the Python packages

    pip install -r requirements.txt

This installs `torch`, `openai-whisper`, `stable-ts` (the forced aligner),
`tqdm` and `imageio-ffmpeg`.

For GPU support install the PyTorch build matching your CUDA version from
https://pytorch.org first, then run the command above. On CPU-only machines the
command above installs CPU PyTorch.

## Usage

### GUI

    python Lryics.py

1. **Audio folder**: the folder with your songs.
2. **Lyrics folder**: where the `.lrc` files are. Leave blank if they sit next
   to the songs (the usual Plex layout).
3. **Output folder**: where converted files go. Leave blank to overwrite in
   place; each original is kept as `<name>.lrc.bak`.
4. Pick a Whisper model and language, tick "Include sub-folders" for a whole
   library, then press **Start Conversion**. **Stop** finishes the current file
   and halts.

The first run downloads the Whisper model.

### Command line

    python Lryics.py --audio "D:/Music" --lyrics "D:/Lyrics" --output "D:/Lyrics/word" --model base --recursive

| Option | Meaning |
| --- | --- |
| `--audio DIR` | folder with the audio files (required for CLI mode) |
| `--lyrics DIR` | folder with the `.lrc` files (default: the audio folder) |
| `--output DIR` | output folder (default: in place, originals kept as `.lrc.bak`) |
| `--model NAME` | `tiny`, `base`, `small`, `medium`, `large`, `large-v3`, `turbo` (default `base`), or the path to a downloaded `.pt` model file |
| `--language CODE` | lyric language, or `auto` to detect per song (default `en`) |
| `--device cpu|cuda` | force a device (default: auto) |
| `-r`, `--recursive` | include sub-folders; the folder structure is mirrored in the output |
| `--reconvert` | also redo files this tool converted before (recognised by their `[re:lrc-align ...]` tag), from the `.lrc.bak` original; word-by-word files from any other source are still left alone. Use after updating the tool. |

### Matching lyrics to audio

A lyric file is matched to the audio file with the same name (`Song.lrc` ->
`Song.mp3`). Matching ignores case, extra spaces and `#U2019`-style escapes
that some zip tools write for characters such as `’`. Lyric files with no
matching audio are skipped and listed in the log.

## Tips

- `base` is a good balance of speed and accuracy. `small` or `medium` place
  words more precisely on busy mixes but are slower; `turbo` is fast on a GPU.
- Running the tool again on the same folders is safe: files that are already
  word-by-word, including previous output, are skipped.
- Loud instrumentals make alignment harder. Lines that fell back to even
  spacing are counted in the log so you can check them.
- Each line is aligned together with the lines just before and after it, so
  its first and last words are not sitting at the edge of the audio slice
  where Whisper tends to misplace them.
- Line timestamps in lyric files are often a few tenths of a second early, so
  the player flips to the next line while the last word is still being sung.
  When a file as a whole runs early, the tool moves each early line's
  timestamp later to where its first word is actually sung, by at most
  0.35 s. Timestamps are never moved earlier, and files whose timestamps
  already match the vocals are left exactly as they are. Every word of a
  line is kept before the next line's timestamp so nothing gets skipped.
  Untick the option in the GUI or pass `--keep-line-times` to leave line
  timestamps exactly as they were.
- When a lyric file was timed to a different edit of the song (every line is
  off by the same amount, common with lyrics from a service and audio from
  elsewhere), the whole file is shifted by that amount and the file is
  flagged in the report.
- Every converted file starts with a provenance tag, the standard LRC id tag
  for the program that wrote the file:

      [re:lrc-align 1.0.0 align word]

  Players ignore it; tools like Beetdrop read it as writer, version, timing
  source and timing level. Any `[re:...]` tag already in the input is
  replaced, since it would otherwise claim the file is still line-level.
  Where the old tag named where the words came from, that is kept as a fifth
  field (`from-apple`). Skipped files are never touched, so re-running the
  tool changes nothing.
- A report named `lrc_conversion_report.txt` is written to the output folder
  after each run. It lists the files worth checking by hand first: files
  that were shifted, lines that could not be aligned, lyric files with no
  matching audio, and failures. Then it lists every file with what was done.
- A line's words are never allowed to spread further than the song's own
  pace justifies, so a held last note before an instrumental break cannot
  drag words into the break. Slow ballads keep their long lines because the
  limit scales with how slowly the song is sung.
- Measured against professionally timed word-by-word files with the `base`
  model: clean pop and country land within 0.5 s for 96 to 99% of words
  (mean error 0.12 to 0.18 s), slow ballads 72 to 93% (mean 0.25 to 0.41 s),
  and screamed hard rock about 60% (mean 0.47 s). Files in that last group
  get a low-confidence flag in the report; the `small` or `medium` model
  does noticeably better on them.
- Speed: on a CPU the `base` model takes roughly a minute for a five minute
  song with dense lyrics. A GPU is many times faster.

## Troubleshooting

### Slow model download

You can manually download the model from the URLs in
https://github.com/openai/whisper/blob/main/whisper/__init__.py and put it in

    Windows: C:\Users\<username>\.cache\whisper\<model>.pt
    Linux:   /home/<username>/.cache/whisper/<model>.pt

### "ffmpeg was not found"

Install ffmpeg and add it to PATH, or run `pip install imageio-ffmpeg`.

## Development

The parsing, timing and file matching logic lives in `lrc_align.py` and has no
GUI or torch dependency, so the tests run without a model:

    python -m unittest discover -s tests
