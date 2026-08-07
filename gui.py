import json
import os
import queue
import threading
import time

import customtkinter as ctk
import requests
from faster_whisper import WhisperModel, format_timestamp

WHISPER_MODELS = ["medium", "large-v3", "small", "base", "tiny"]
DEFAULT_WHISPER = "medium"

FREE_MODELS_FALLBACK = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# ---------- Palette ----------
BG      = "#0e1117"
PANEL   = "#131722"
PANEL_2 = "#1b2130"
CARD    = "#171c28"
BORDER  = "#262d3d"
FG      = "#e8ecf4"
MUTED   = "#8b95a7"
ACCENT  = "#4f8cff"
ACCENT_D = "#3465c9"
GREEN   = "#2ecc71"
RED     = "#ff5c5c"


class StopRequested(Exception):
    pass


class FreeModelCombo(ctk.CTkComboBox):
    """CTkComboBox that live-fetches OpenRouter free models every time
    the dropdown is opened (no hardcoded list)."""

    def __init__(self, *args, on_open=None, **kwargs):
        self._on_open = on_open
        super().__init__(*args, **kwargs)

    def _open_dropdown_menu(self):
        if callable(self._on_open):
            self._on_open()
        super()._open_dropdown_menu()


class App:

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Movie Explain - Video Highlights")
        self.root.geometry("1060x700")
        self.root.minsize(960, 640)

        self.msg_queue = queue.Queue()
        self.running = False

        self.video_path = ctk.StringVar(value="video.mp4")
        self.prompt_path = ctk.StringVar(value="prompt.txt")
        self.whisper_model = ctk.StringVar(value=DEFAULT_WHISPER)
        self.language = ctk.StringVar(value="bn")
        self.ai_model = ctk.StringVar(value=DEFAULT_MODEL)
        self.status_text = ctk.StringVar(value="Idle")

        self._load_api_key()
        self._build_ui()
        self._poll_queue()
        self._fetch_models_async("startup")

    # ----------------------------------------------------------------
    # Config / API keys
    # ----------------------------------------------------------------

    def _load_api_key(self):
        try:
            from config import OPENROUTER_API_KEY
        except Exception:
            OPENROUTER_API_KEY = ""
        self.openrouter_key = (OPENROUTER_API_KEY or "").strip()

    # ----------------------------------------------------------------
    # UI build
    # ----------------------------------------------------------------

    def _build_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        outer.pack(fill="both", expand=True)

        self._build_header(outer)

        body = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=(4, 10))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, fg_color=BG, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.grid_columnconfigure(0, weight=1)

        self._card_source(left)
        self._card_transcription(left)
        self._card_ai(left)

        right = ctk.CTkFrame(body, fg_color=BG, corner_radius=0, width=430)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        self._build_run_card(right)
        self._build_console(right)

        self._build_footer()

    # ---- header ----
    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, height=76)
        header.pack(fill="x", side="top")
        header.grid_columnconfigure(0, weight=1)

        logo = ctk.CTkFrame(header, fg_color=ACCENT, width=44, height=44, corner_radius=11)
        logo.pack(side="left", padx=(24, 14), pady=14)
        logo.pack_propagate(False)
        ctk.CTkLabel(logo, text="⏵", text_color="#0e1117", font=ctk.CTkFont(size=22, weight="bold")).pack(expand=True)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(title_box, text="Movie Explain", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=FG).pack(anchor="w")
        ctk.CTkLabel(title_box, text="Auto extract video highlights with Whisper + AI",
                     font=ctk.CTkFont(size=12), text_color=MUTED).pack(anchor="w")

        header_pill = ctk.CTkLabel(header, text="   v1.0   ", font=ctk.CTkFont(size=11),
                                   fg_color=PANEL_2, text_color=MUTED, corner_radius=10, height=24)
        header_pill.pack(side="right", padx=20)

    # ---- card helpers ----
    def _section_title(self, parent, icon, title, row):
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.grid(row=row, column=0, sticky="ew", padx=16, pady=(16, 4))
        ctk.CTkLabel(head, text=icon, font=ctk.CTkFont(size=14), text_color=ACCENT).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color=FG).pack(side="left")
        ctk.CTkFrame(head, fg_color=BORDER, height=1).pack(fill="x", pady=(10, 0))

    def _card_source(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(0, weight=1)
        self._section_title(card, "🎬", "Source Files", 0)
        self._file_row(card, "Video File", self.video_path, "video", 1)
        self._file_row(card, "Prompt", self.prompt_path, "prompt", 2)

    def _file_row(self, parent, label, var, kind, row):
        row_f = ctk.CTkFrame(parent, fg_color="transparent")
        row_f.grid(row=row, column=0, sticky="ew", padx=16, pady=(10, 4))
        row_f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row_f, text=label, font=ctk.CTkFont(size=13), text_color=MUTED,
                     width=66, anchor="w").grid(row=0, column=0, sticky="w")
        entry = ctk.CTkEntry(row_f, textvariable=var, font=ctk.CTkFont(size=13), fg_color=PANEL_2,
                             border_color=BORDER, corner_radius=8, height=36)
        entry.grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(row_f, text="Browse", width=78, height=36, corner_radius=8, font=ctk.CTkFont(size=13),
                      fg_color=PANEL_2, hover_color="#283145", text_color=FG, border_width=1,
                      border_color=BORDER, command=lambda k=kind: self._browse(k)).grid(row=0, column=2, padx=(8, 0))

    def _card_transcription(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(0, weight=1)
        self._section_title(card, "🎙", "Transcription (Whisper)", 0)

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew", padx=16)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(grid, text="Model", font=ctk.CTkFont(size=13), text_color=MUTED).grid(
            row=0, column=0, sticky="w")
        self.whisper_combo = ctk.CTkComboBox(grid, variable=self.whisper_model, values=WHISPER_MODELS,
                                             fg_color=PANEL_2, border_color=BORDER, button_color=ACCENT,
                                             corner_radius=8, font=ctk.CTkFont(size=13))
        self.whisper_combo.set(self.whisper_model.get())
        self.whisper_combo.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        ctk.CTkLabel(grid, text="Language", font=ctk.CTkFont(size=13), text_color=MUTED).grid(
            row=0, column=2, sticky="w")
        ctk.CTkEntry(grid, textvariable=self.language, fg_color=PANEL_2, border_color=BORDER,
                     corner_radius=8, font=ctk.CTkFont(size=13), height=32).grid(
            row=1, column=2, sticky="ew", pady=(4, 0))

        ctk.CTkLabel(card, text="(blank = auto-detect · e.g. bn, hi, en)",
                     font=ctk.CTkFont(size=12), text_color=MUTED).grid(row=2, column=0, sticky="w",
                                                                       padx=16, pady=(4, 12))
        foot = ctk.CTkFrame(card, fg_color="transparent")
        foot.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 14))
        ctk.CTkButton(foot, text="Warm-up Model", width=140, height=34, corner_radius=8,
                      font=ctk.CTkFont(size=13), fg_color=PANEL_2, hover_color="#334155",
                      text_color=FG, border_width=1, border_color=BORDER,
                      command=self._load_whisper).pack(side="left")
        ctk.CTkLabel(foot, text="pre-loads the model to skip first-run delay",
                     font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left", padx=(10, 0))

    def _card_ai(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=2, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        self._section_title(card, "🧠", "AI Explanations (OpenRouter)", 0)

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 4))
        grid.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(grid, text="Free AI Model", font=ctk.CTkFont(size=13), text_color=MUTED).grid(
            row=0, column=0, sticky="w")
        model_row = ctk.CTkFrame(grid, fg_color="transparent")
        model_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        model_row.grid_columnconfigure(0, weight=1)
        self.ai_model_combo = FreeModelCombo(model_row, variable=self.ai_model,
                                             values=FREE_MODELS_FALLBACK, on_open=self._on_model_menu_open,
                                             fg_color=PANEL_2, border_color=BORDER, button_color=ACCENT,
                                             corner_radius=8, font=ctk.CTkFont(size=13))
        self.ai_model_combo.set(self.ai_model.get())
        self.ai_model_combo.grid(row=0, column=0, sticky="ew")
        self.model_refresh_btn = ctk.CTkButton(model_row, text="⟳", width=44, height=36, corner_radius=8,
                                               font=ctk.CTkFont(size=15, weight="bold"), fg_color=PANEL_2,
                                               hover_color="#334155", text_color=FG, border_width=1,
                                               border_color=BORDER, command=self._refresh_models_now)
        self.model_refresh_btn.grid(row=0, column=1, padx=(8, 0))

        ctk.CTkLabel(card, text="Live list from OpenRouter — tap the dropdown to refresh. (free = 0.00 $)",
                     font=ctk.CTkFont(size=12), text_color=MUTED).grid(row=2, column=0, sticky="w",
                                                                       padx=16, pady=(4, 14))

    # ---- run card (right column) ----
    def _build_run_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        self.pb = ctk.CTkProgressBar(card, fg_color=PANEL_2, progress_color=ACCENT, corner_radius=6, height=12)
        self.pb.set(0)
        ctk.CTkLabel(card, text="Pipeline", font=ctk.CTkFont(size=14, weight="bold"), text_color=FG
                     ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 2))
        self._add_progress_dots(card)
        self.pb.grid(row=3, column=0, sticky="ew", padx=16, pady=(10, 4))

        self.run_btn = ctk.CTkButton(card, text="▶  Start Extraction", height=52, corner_radius=12,
                                     font=ctk.CTkFont(size=15, weight="bold"), fg_color=ACCENT,
                                     hover_color=ACCENT_D, command=self._run)
        self.run_btn.grid(row=4, column=0, sticky="ew", padx=16, pady=(8, 0))

        self.status_dot = ctk.CTkLabel(card, text="●", font=ctk.CTkFont(size=16), text_color=GREEN)
        self.status_dot.grid(row=5, column=0, pady=(14, 0))
        ctk.CTkLabel(card, textvariable=self.status_text, font=ctk.CTkFont(size=13),
                     text_color=MUTED).grid(row=6, column=0, pady=(0, 14))

    def _add_progress_dots(self, parent):
        step_bar = ctk.CTkFrame(parent, fg_color="transparent")
        step_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 0))
        for i in range(5):
            ctk.CTkFrame(step_bar, fg_color=BORDER, width=8, height=8, corner_radius=4,
                         ).pack(side="left", padx=3, fill="both")
        step_names = ctk.CTkFrame(parent, fg_color="transparent")
        step_names.grid(row=2, column=0, sticky="ew", padx=16)
        for txt in ["Model", "Transcribe", "Srt", "AI", "Save"]:
            ctk.CTkLabel(step_names, text=txt, font=ctk.CTkFont(size=10), text_color=MUTED).pack(side="left", expand=True)

    # ---- console ----
    def _build_console(self, parent):
        title = ctk.CTkFrame(parent, fg_color="transparent")
        title.grid(row=1, column=0, sticky="ew", pady=(16, 6))
        ctk.CTkLabel(title, text="Console", font=ctk.CTkFont(size=14, weight="bold"), text_color=FG).pack(side="left")
        ctk.CTkButton(title, text="Clear", width=64, height=28, corner_radius=8, font=ctk.CTkFont(size=12),
                      fg_color=PANEL_2, hover_color="#334155", text_color=FG, border_width=1,
                      border_color=BORDER, command=self._clear_log).pack(side="right")

        self.console = ctk.CTkTextbox(parent, font=ctk.CTkFont(family="monospace", size=13), fg_color="#0a0d13",
                                      text_color="#cfe3f5", corner_radius=10, border_width=1,
                                      border_color=BORDER, wrap="word")
        self.console.grid(row=2, column=0, sticky="nsew")

        self.console.tag_config("info", foreground="#dde6f5")
        self.console.tag_config("success", foreground="#7ddb9a")
        self.console.tag_config("error", foreground="#ff8080")
        self.console.insert("end", "Ready.  Load a video and press Start.\n", "info")
        self.console.configure(state="disabled")

    def _build_footer(self):
        footer = ctk.CTkFrame(self.root, fg_color=PANEL, corner_radius=0, height=40)
        footer.pack(fill="x", side="bottom")
        tip = "   Tip: OpenRouter offers free models. Keep the model name exactly as shown."
        ctk.CTkLabel(footer, text=tip, font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(footer, text="Movie Explain", font=ctk.CTkFont(size=12),
                     text_color=MUTED).pack(side="right", padx=16)

    # ----------------------------------------------------------------
    # UI actions
    # ----------------------------------------------------------------

    def _browse(self, kind):
        from tkinter import filedialog
        if kind == "video":
            path = filedialog.askopenfilename(title="Select video",
                                              filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
            if path:
                self.video_path.set(path)
        else:
            path = filedialog.askopenfilename(title="Select prompt",
                                              filetypes=[("Text", "*.txt"), ("All files", "*")])
            if path:
                self.prompt_path.set(path)

    # ----------------------------------------------------------------
    # Live OpenRouter free-model list
    # ----------------------------------------------------------------

    def _fetch_free_models(self, timeout=6):
        try:
            resp = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
            resp.raise_for_status()
            free = []
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                pricing = m.get("pricing") or {}
                if (pricing.get("prompt") == "0" and pricing.get("completion") == "0"
                        and mid and "embed" not in mid):
                    free.append(mid)
            return sorted(set(free)) or None
        except Exception:
            return None

    def _apply_models(self, models, _source):
        if not models:
            return False
        current = self.ai_model.get()
        self.ai_model_combo.configure(values=models)
        self.ai_model.set(current if current in models else models[0])
        self._model_count = len(models)
        return True

    def _fetch_models_async(self, source):
        def job():
            models = self._fetch_free_models()
            def done():
                self._apply_models(models, source)
                if models:
                    self.log(f"Loaded {len(models)} free models from OpenRouter.", "info")
            self.root.after(0, done)
        threading.Thread(target=job, daemon=True).start()

    def _on_model_menu_open(self):
        models = self._fetch_free_models(timeout=4)
        if models:
            self._apply_models(models, "live")

    def _refresh_models_now(self):
        self.model_refresh_btn.configure(state="disabled", text="…")
        def job():
            models = self._fetch_free_models()
            def done():
                self.model_refresh_btn.configure(state="normal", text="⟳")
                if models and self._apply_models(models, "refresh"):
                    self.log(f"Refreshed: {len(models)} free models from OpenRouter.", "success")
                else:
                    self.log("Model refresh failed — no internet?", "error")
            self.root.after(0, done)
        threading.Thread(target=job, daemon=True).start()

    def _clear_log(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("end", "Log cleared.\n", "info")
        self.console.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                message, level, ts = self.msg_queue.get_nowait()
                tag = "info" if level == "info" else level
                self.console.configure(state="normal")
                self.console.insert("end", f"  {ts}  {message}\n", tag)
                self.console.configure(state="disabled")
                self.console.see("end")
                if level == "error":
                    self.status_text.set(f"Error: {message}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def log(self, message, level="info"):
        self.msg_queue.put((message, level, time.strftime("%H:%M:%S")))

    # ----------------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------------

    def _run(self):
        if self.running:
            return
        if not os.path.exists(self.video_path.get()):
            from tkinter import messagebox
            messagebox.showerror("Error", "Video file not found:\n" + self.video_path.get())
            return
        if not os.path.exists(self.prompt_path.get()):
            from tkinter import messagebox
            messagebox.showerror("Error", "Prompt file not found:\n" + self.prompt_path.get())
            return
        if not self.openrouter_key:
            from tkinter import messagebox
            messagebox.showerror("Error", "OpenRouter API key not found.\n\n"
                                "Set OPENROUTER_API_KEY in config.py.")
            return

        self.log("── Starting extraction ──", "info")
        self.status_text.set("Running...")
        self.running = True
        self.run_btn.configure(text="■  Stop", fg_color="#b03a2e", hover_color="#8f2d24")
        self.status_dot.configure(text_color=RED)
        self.pb.set(0.05)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        try:
            self._step_transcribe()
            self._step_ai()
            self.log("── Done ──", "success")
            self.log("Outputs saved to video.srt and highlight.json", "success")
        except StopRequested:
            self.log("Pipeline stopped by user.", "info")
        except Exception as e:
            self.log(f"Pipeline failed: {type(e).__name__}: {e}", "error")
        finally:
            self.root.after(0, self._reset_running)

    def _reset_running(self):
        self.running = False
        self.status_text.set("Idle")
        self.status_dot.configure(text_color=GREEN)
        self.run_btn.configure(text="▶  Start Extraction", fg_color=ACCENT, hover_color=ACCENT_D)
        self.pb.set(0)

    # ----------------------------------------------------------------
    # Steps
    # ----------------------------------------------------------------

    def _step_transcribe(self):
        self.log("Loading Whisper model…", "info")
        model = WhisperModel(self.whisper_model.get(), device="cpu", compute_type="int8")

        language = self.language.get().strip() or None
        self.log("Transcribing…  (this can take a while)", "info")
        kwargs = {"language": language} if language else {}
        result = model.transcribe(self.video_path.get(), beam_size=5, task="transcribe", **kwargs)

        segments, info = result
        self.log(f"Detected language: {info.language}", "info")

        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg.start, always_include_hours=True, decimal_marker=",")
            end = format_timestamp(seg.end, always_include_hours=True, decimal_marker=",")
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
            self.log(f"  [{start} → {end}] {seg.text.strip()}", "info")

        with open("video.srt", "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        self.log(f"SRT saved  →  video.srt  ({len(srt_lines)} lines)", "success")
        self.pb.set(0.6)

    def _step_ai(self):
        with open("video.srt", "r", encoding="utf-8") as f:
            subtitle = f.read()
        with open(self.prompt_path.get(), "r", encoding="utf-8") as f:
            prompt = f.read()

        ai_model = self.ai_model.get().strip()
        self.log(f"AI →  OpenRouter  ·  {ai_model}", "info")

        text = self._call_openrouter(self.openrouter_key, ai_model, prompt, subtitle)

        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        with open("highlight.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.log("highlight.json saved  →", "success")
        self.pb.set(1.0)

    @staticmethod
    def _call_openrouter(api_key, ai_model, prompt, subtitle):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": ai_model,
                   "messages": [{"role": "user", "content": f"{prompt}\n\nSubtitle:\n\n{subtitle}"}]}
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]["content"]

    def _load_whisper(self):
        def job():
            self.log("Pre-loading Whisper model…", "info")
            try:
                WhisperModel(self.whisper_model.get(), device="cpu", compute_type="int8")
                self.log("Model ready — first Run will be fast now.", "success")
            except Exception as e:
                self.log(f"Model load failed: {e}", "error")
        threading.Thread(target=job, daemon=True).start()


def main():
    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    main()