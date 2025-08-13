#!/usr/bin/env python3
import subprocess
import platform
import whisper
import torch
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

class LRCBatchTranscriberApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Audio to LRC Transcriber")
        self.geometry("800x600")

        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_text = tk.StringVar(value="Select input and output folders, then start.")

        # UI Setup
        self.create_widgets()
        self.model = None

    def open_output_folder(self):
        path = self.output_dir.get()
        if not path:
            messagebox.showerror("Error", "Output folder path is not set.")
            return
        path = Path(path)
        if not path.exists():
            messagebox.showerror("Error", "Output folder does not exist.")
            return

        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(f'explorer "{path}"')
            elif system == "Linux":
                subprocess.Popen(["xdg-open", str(path)])
            else:
                messagebox.showinfo("Unsupported OS", f"Opening folder not supported on {system}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {e}")

    def create_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Input folder
        ttk.Label(frm, text="Input folder (songs):").pack(anchor=tk.W)
        input_frame = ttk.Frame(frm)
        input_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(input_frame, textvariable=self.input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.LEFT)

        # Output folder
        ttk.Label(frm, text="Output folder (LRC files):").pack(anchor=tk.W)
        output_frame = ttk.Frame(frm)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT)

        # Model selector
        ttk.Label(frm, text="Select Whisper Model:").pack(anchor=tk.W, pady=(10,4))
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(frm, textvariable=self.model_var, state="readonly",
                                   values=["tiny", "base", "small", "medium", "large"])
        model_combo.pack(fill=tk.X, pady=(0, 15))

        # Start button
        self.start_btn = ttk.Button(frm, text="Start Transcription", command=self.start_transcription)
        self.start_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.pack(fill=tk.X, pady=5)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(frm, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status label
        self.status_label = ttk.Label(frm, textvariable=self.status_text)
        self.status_label.pack(anchor=tk.W, pady=2)

        # Open output folder button
        self.open_folder_btn = ttk.Button(frm, text="Open Output Folder", command=self.open_output_folder)
        self.open_folder_btn.pack(pady=5, anchor=tk.E)
        self.open_folder_btn.pack_forget()  # Hide initially

    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_dir.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir.set(folder)

    def log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def start_transcription(self):
        input_path = self.input_dir.get()
        output_path = self.output_dir.get()
        if not input_path or not output_path:
            messagebox.showerror("Error", "Please select both an input and output folder.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress.start()
        self.status_text.set("Loading model...")
        threading.Thread(target=self.transcribe_folder, args=(Path(input_path), Path(output_path)), daemon=True).start()

    def transcribe_folder(self, input_dir, output_dir):
        try:
            # Find audio files
            audio_extensions = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]
            files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in audio_extensions])
            if not files:
                self.log("No audio files found in input folder!")
                self.status_text.set("No audio files found.")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_choice = self.model_var.get()
            self.log(f"Loading Whisper model '{model_choice}' on {device}...")
            self.model = whisper.load_model(model_choice).to(device)

            self.progress['maximum'] = len(files)

            for idx, file in enumerate(files, 1):
                self.log(f"Transcribing: {file.name}")
                result = self.model.transcribe(str(file))

                # Build LRC lines
                lrc_lines = []
                for seg in result['segments']:
                    start_time = seg['start']
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    hundredths = int((start_time - int(start_time)) * 100)
                    timestamp = f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]"
                    lrc_lines.append(f"{timestamp} {seg['text'].strip()}")

                # Save with same base name
                output_file = output_dir / (file.stem + ".lrc")
                output_file.write_text("\n".join(lrc_lines), encoding="utf-8")
                self.log(f"Saved: {output_file.name}")

                self.progress['value'] = idx
                self.status_text.set(f"Processed {idx}/{len(files)} files")

            self.status_text.set("All files processed!")
            self.open_folder_btn.pack()
        except Exception as e:
            self.log(f"Error: {e}")
            self.status_text.set("Error occurred.")
        finally:
            self.start_btn.config(state=tk.NORMAL)
            self.progress.stop()

if __name__ == "__main__":
    app = LRCBatchTranscriberApp()
    app.mainloop()
