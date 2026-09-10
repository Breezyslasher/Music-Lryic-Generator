#!/usr/bin/env python3
"""Batch converter: line-by-line LRC lyrics -> word-by-word (enhanced) LRC.

Run without arguments for the GUI, or pass ``--audio`` for command line use:

    python Lryics.py --audio "D:/Music" --lyrics "D:/Music" --output "D:/Music/word-lrc"

Every ``.lrc`` in the lyrics folder is matched to the audio file with the same
name.  Lines are force-aligned to the audio to get a timestamp for each word.
Files that are already word-by-word are skipped.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import threading
from pathlib import Path

import lrc_align

MODELS = ["tiny", "base", "small", "medium", "large", "large-v3", "turbo"]
LANGUAGES = ["en", "auto", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "vi", "ru", "nl", "sv"]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def run_cli(args: argparse.Namespace) -> int:
    audio_dir = Path(args.audio)
    lyrics_dir = Path(args.lyrics) if args.lyrics else audio_dir
    output_dir = Path(args.output) if args.output else lyrics_dir
    if not audio_dir.is_dir():
        print(f"Audio folder not found: {audio_dir}", file=sys.stderr)
        return 2
    if not lyrics_dir.is_dir():
        print(f"Lyrics folder not found: {lyrics_dir}", file=sys.stderr)
        return 2

    only = None
    reconvert = args.reconvert
    if args.only_flagged is not None:
        report = Path(args.only_flagged) if args.only_flagged else output_dir / lrc_align.REPORT_NAME
        if not report.is_file():
            print(f"Report not found: {report}", file=sys.stderr)
            return 2
        only = lrc_align.flagged_names_from_report(report)
        reconvert = True
        print(f"{len(only)} flagged file(s) listed in {report.name}")
        if not only:
            return 0

    print(f"Loading Whisper model '{args.model}'...")
    aligner = lrc_align.WhisperLineAligner(args.model, device=args.device)
    print(f"Model loaded on {aligner.device}.")
    summary = lrc_align.process_library(
        audio_dir, lyrics_dir, output_dir, aligner,
        language=args.language, recursive=args.recursive, log=print,
        retime_lines=not args.keep_line_times, reconvert=reconvert, only=only,
    )
    print("Done: " + summary.describe())
    return 0 if summary.count("failed") == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert line-by-line LRC files to word-by-word LRC using the audio.")
    p.add_argument("--audio", help="Folder containing the audio files (starts the GUI when omitted)")
    p.add_argument("--lyrics", help="Folder containing the .lrc files (default: same as --audio)")
    p.add_argument("--output", help="Folder to write converted .lrc files (default: in place, originals kept as .lrc.bak)")
    p.add_argument("--model", default="base",
                   help="Whisper model name (" + ", ".join(MODELS) + ") or path to a downloaded .pt file")
    p.add_argument("--language", default="en", help="Lyric language code, or 'auto' to detect per song")
    p.add_argument("--device", default=None, help="Force 'cpu' or 'cuda' (default: auto)")
    p.add_argument("-r", "--recursive", action="store_true", help="Search sub-folders too")
    p.add_argument("--reconvert", action="store_true",
                   help="Also redo files this tool converted before (from their .lrc.bak); "
                        "word-by-word files from other sources are still left alone")
    p.add_argument("--only-flagged", metavar="REPORT", nargs="?", const="",
                   help="Only process the files listed as flagged in a report from an earlier run "
                        "(default: the report in the output folder). Implies --reconvert.")
    p.add_argument("--keep-line-times", action="store_true",
                   help="Keep the original line timestamps instead of moving them to the sung first word")
    return p


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    class LRCWordAlignerApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("LRC Line-to-Word Converter")
            self.geometry("820x640")

            self.audio_dir = tk.StringVar()
            self.lyrics_dir = tk.StringVar()
            self.output_dir = tk.StringVar()
            self.model_var = tk.StringVar(value="base")
            self.language_var = tk.StringVar(value="en")
            self.recursive_var = tk.BooleanVar(value=False)
            self.retime_var = tk.BooleanVar(value=True)
            self.reconvert_var = tk.BooleanVar(value=False)
            self.only_flagged_var = tk.BooleanVar(value=False)
            self.report_path = tk.StringVar()
            self.status_text = tk.StringVar(
                value="Pick the folder with your songs. Lyrics/output default to the same folder.")
            self.stop_event = threading.Event()
            self.worker = None

            self.create_widgets()

        # ---- layout -------------------------------------------------------
        def folder_row(self, parent, label, var, browse):
            ttk.Label(parent, text=label).pack(anchor=tk.W)
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(row, text="Browse", command=browse).pack(side=tk.LEFT)

        def create_widgets(self):
            frm = ttk.Frame(self, padding=10)
            frm.pack(fill=tk.BOTH, expand=True)

            self.folder_row(frm, "Audio folder (songs):", self.audio_dir, self.browse_audio)
            self.folder_row(frm, "Lyrics folder (.lrc files, blank = audio folder):", self.lyrics_dir,
                            self.browse_lyrics)
            self.folder_row(frm, "Output folder (blank = overwrite in place, originals kept as .lrc.bak):",
                            self.output_dir, self.browse_output)

            opts = ttk.Frame(frm)
            opts.pack(fill=tk.X, pady=(10, 4))
            ttk.Label(opts, text="Whisper model (name or path to .pt):").pack(side=tk.LEFT)
            ttk.Combobox(opts, textvariable=self.model_var, values=MODELS, width=14
                         ).pack(side=tk.LEFT, padx=(4, 16))
            ttk.Label(opts, text="Language:").pack(side=tk.LEFT)
            ttk.Combobox(opts, textvariable=self.language_var, values=LANGUAGES, width=8
                         ).pack(side=tk.LEFT, padx=(4, 16))
            ttk.Checkbutton(opts, text="Include sub-folders", variable=self.recursive_var).pack(side=tk.LEFT)
            ttk.Checkbutton(frm, text="Move line timestamps to where the first word is sung (recommended)",
                            variable=self.retime_var).pack(anchor=tk.W, pady=(0, 4))
            ttk.Checkbutton(frm, text="Re-do files this tool converted before (uses the .lrc.bak originals)",
                            variable=self.reconvert_var).pack(anchor=tk.W, pady=(0, 4))
            flagged = ttk.Frame(frm)
            flagged.pack(fill=tk.X, pady=(0, 4))
            ttk.Checkbutton(flagged, text="Only files flagged in a report (redone from .lrc.bak, e.g. with a bigger model):",
                            variable=self.only_flagged_var).pack(side=tk.LEFT)
            ttk.Entry(flagged, textvariable=self.report_path, width=28).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                              padx=(4, 0))
            ttk.Button(flagged, text="Browse", command=self.browse_report).pack(side=tk.LEFT)
            ttk.Label(frm, text="    (leave the report path blank to use the report in the output folder)",
                      foreground="gray").pack(anchor=tk.W)

            btns = ttk.Frame(frm)
            btns.pack(pady=10)
            self.start_btn = ttk.Button(btns, text="Start Conversion", command=self.start)
            self.start_btn.pack(side=tk.LEFT, padx=4)
            self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state=tk.DISABLED)
            self.stop_btn.pack(side=tk.LEFT, padx=4)

            self.progress = ttk.Progressbar(frm, mode="determinate")
            self.progress.pack(fill=tk.X, pady=5)

            self.log_text = scrolledtext.ScrolledText(frm, height=14, state=tk.DISABLED)
            self.log_text.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frm, textvariable=self.status_text).pack(anchor=tk.W, pady=2)

            self.open_folder_btn = ttk.Button(frm, text="Open Output Folder", command=self.open_output_folder)
            self.open_folder_btn.pack(pady=5, anchor=tk.E)
            self.open_folder_btn.pack_forget()

        # ---- helpers ------------------------------------------------------
        def browse_audio(self):
            folder = filedialog.askdirectory(title="Select Audio Folder")
            if folder:
                self.audio_dir.set(folder)

        def browse_lyrics(self):
            folder = filedialog.askdirectory(title="Select Lyrics Folder")
            if folder:
                self.lyrics_dir.set(folder)

        def browse_report(self):
            path = filedialog.askopenfilename(title="Select a conversion report",
                                              filetypes=[("Report", "*.txt"), ("All files", "*")])
            if path:
                self.report_path.set(path)
                self.only_flagged_var.set(True)

        def browse_output(self):
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_dir.set(folder)

        def log(self, message):
            # Called from the worker thread; hand the update to the Tk thread.
            self.after(0, self._append_log, message)

        def _append_log(self, message):
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        def set_status(self, message):
            self.after(0, self.status_text.set, message)

        def set_progress(self, done, total):
            def apply():
                self.progress.configure(maximum=total, value=done)
                self.status_text.set(f"Processed {done}/{total} files")
            self.after(0, apply)

        def open_output_folder(self):
            path = self.output_dir.get() or self.lyrics_dir.get() or self.audio_dir.get()
            if not path or not Path(path).exists():
                messagebox.showerror("Error", "Output folder does not exist.")
                return
            system = platform.system()
            try:
                if system == "Windows":
                    subprocess.Popen(["explorer", str(path)])
                elif system == "Darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to open folder: {e}")

        # ---- run ----------------------------------------------------------
        def start(self):
            audio = self.audio_dir.get().strip()
            if not audio:
                messagebox.showerror("Error", "Please select the audio folder.")
                return
            lyrics = self.lyrics_dir.get().strip() or audio
            output = self.output_dir.get().strip() or lyrics
            for label, folder in (("Audio", audio), ("Lyrics", lyrics)):
                if not Path(folder).is_dir():
                    messagebox.showerror("Error", f"{label} folder does not exist:\n{folder}")
                    return
            if Path(output).resolve() == Path(lyrics).resolve():
                if not messagebox.askyesno(
                        "Overwrite in place?",
                        "Converted files will replace the originals in the lyrics folder.\n"
                        "Each original is kept as <name>.lrc.bak. Continue?"):
                    return

            only = None
            if self.only_flagged_var.get():
                report = Path(self.report_path.get().strip() or Path(output) / lrc_align.REPORT_NAME)
                if not report.is_file():
                    messagebox.showerror("Error", f"Report not found:\n{report}")
                    return
                only = lrc_align.flagged_names_from_report(report)
                if not only:
                    messagebox.showinfo("Nothing flagged", f"No flagged files listed in {report.name}.")
                    return

            self.stop_event.clear()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.open_folder_btn.pack_forget()
            self.progress.configure(value=0)
            self.status_text.set("Loading model...")
            self.worker = threading.Thread(
                target=self.run, args=(Path(audio), Path(lyrics), Path(output), only), daemon=True)
            self.worker.start()

        def stop(self):
            self.stop_event.set()
            self.set_status("Stopping after the current file...")
            self.stop_btn.config(state=tk.DISABLED)

        def run(self, audio_dir, lyrics_dir, output_dir, only=None):
            try:
                model = self.model_var.get()
                if only is not None:
                    self.log(f"Limiting this run to {len(only)} flagged file(s).")
                self.log(f"Loading Whisper model '{model}'...")
                aligner = lrc_align.WhisperLineAligner(model)
                self.log(f"Model loaded on {aligner.device}.")
                self.set_status("Converting...")
                summary = lrc_align.process_library(
                    audio_dir, lyrics_dir, output_dir, aligner,
                    language=self.language_var.get().strip() or "en",
                    recursive=self.recursive_var.get(),
                    log=self.log, progress=self.set_progress,
                    should_stop=self.stop_event.is_set,
                    retime_lines=self.retime_var.get(),
                    reconvert=self.reconvert_var.get() or only is not None,
                    only=only,
                )
                self.log("Done: " + summary.describe())
                self.set_status("Finished: " + summary.describe())
                self.after(0, self.open_folder_btn.pack)
            except Exception as e:  # noqa: BLE001
                self.log(f"Error: {e}")
                self.set_status("Error occurred.")
            finally:
                self.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    LRCWordAlignerApp().mainloop()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.audio:
        return run_cli(args)
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
