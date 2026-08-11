import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import customtkinter as ctk
import requests
from faster_whisper import WhisperModel, format_timestamp

WHISPER_MODELS = ["large-v3", "medium", "small", "base", "tiny"]
DEFAULT_WHISPER = "large-v3"

FREE_MODELS_FALLBACK = [
    "nvidia/nemotron-3-nano-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-4b:free",
]
DEFAULT_MODEL = FREE_MODELS_FALLBACK[0]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
PROJECTS_DIR = "projects"
LAST_PROJECT_FILE = os.path.join(PROJECTS_DIR, ".last_project")

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


class RateLimitedError(Exception):
    pass


class InvalidModelError(Exception):
    pass


class AiErrorResponse(Exception):
    pass


TOLERANT_HINT = ("\n\nIf the transcript is too noisy to understand, still return your best"
                 " guess of 1-3 likely highlight intervals from the timestamps."
                 " NEVER respond with an error object, explanation, or any text."
                 " ONLY return the JSON object with the clips array.")


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
        self.cutting = False
        self.highlights = []
        self.project_name = ""
        self.stages = {"transcribe": False, "ai": False, "cut": False}

        self.video_path = ctk.StringVar(value="video.mp4")
        self.prompt_path = ctk.StringVar(value="prompt.txt")
        self.whisper_model = ctk.StringVar(value=DEFAULT_WHISPER)
        self.language = ctk.StringVar(value="")
        self.ai_model = ctk.StringVar(value=DEFAULT_MODEL)
        self.status_text = ctk.StringVar(value="Idle")

        self._load_api_key()
        self._build_ui()
        self._poll_queue()
        self._fetch_models_async("startup")
        self._load_highlights()
        self._restore_last_project()

    # ----------------------------------------------------------------
    # Project system (CapCut-style: folder per project, resume anytime)
    # ----------------------------------------------------------------

    def _projects_dir(self):
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        return PROJECTS_DIR

    def _project_folder(self):
        if not self.project_name:
            return None
        return os.path.join(self._projects_dir(), self.project_name)

    def _project_file(self, fname):
        folder = self._project_folder()
        return os.path.join(folder, fname) if folder else fname

    def _project_list(self):
        d = self._projects_dir()
        names = [n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)) and not n.startswith(".")]
        return sorted(names, key=str.lower)

    def _save_project(self):
        folder = self._project_folder()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        state = {
            "name": self.project_name,
            "source_video": self.video_path.get(),
            "prompt": self.prompt_path.get(),
            "whisper_model": self.whisper_model.get(),
            "language": self.language.get(),
            "ai_model": self.ai_model.get(),
            "stages": self.stages,
            "updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(os.path.join(folder, "project.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            with open(LAST_PROJECT_FILE, "w", encoding="utf-8") as f:
                f.write(self.project_name)
        except OSError:
            pass

    def _load_project(self, name):
        name = os.path.basename(str(name).strip())
        folder = os.path.join(self._projects_dir(), name)
        if not os.path.isdir(folder):
            self.log(f"Project not found: {name}", "error")
            return
        self.project_name = name
        try:
            with open(os.path.join(folder, "project.json"), encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            state = {}
        self.video_path.set(state.get("source_video") or "")
        if state.get("prompt"):
            self.prompt_path.set(state["prompt"])
        if state.get("whisper_model"):
            self.whisper_model.set(state["whisper_model"])
        for key in ("ai_model", "language"):
            if state.get(key):
                getattr(self, key).set(state[key])
        self.stages = {
            "transcribe": bool(state.get("stages", {}).get("transcribe")),
            "ai": bool(state.get("stages", {}).get("ai")),
            "cut": bool(state.get("stages", {}).get("cut")),
        }
        self._refresh_project_menu()
        self._update_stage_chips()
        self._load_highlights()
        if not os.path.exists(self.video_path.get()):
            self.log(f"Note: saved video not found ({self.video_path.get()}) — browse to re-link it.", "error")
        else:
            self.log(f"Project opened: {name} — resume at stage {self._stage_summary()}", "success")

    def _restore_last_project(self):
        name = None
        try:
            if os.path.exists(LAST_PROJECT_FILE):
                with open(LAST_PROJECT_FILE, encoding="utf-8") as f:
                    name = f.read().strip()
        except OSError:
            pass
        if name and os.path.isdir(os.path.join(self._projects_dir(), name)):
            self._load_project(name)

    def _new_project(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("New Project", "Project name:", parent=self.root)
        if not name:
            return
        name = os.path.basename(name.strip())
        if not name:
            return
        folder = os.path.join(self._projects_dir(), name)
        os.makedirs(folder, exist_ok=True)
        self.project_name = name
        self.stages = {"transcribe": False, "ai": False, "cut": False}
        self._save_project()
        self._refresh_project_menu()
        self._update_stage_chips()
        self.log(f"New project created: {name}", "success")
        self.log("Choose a video, then press Run. You can close the app anytime — "
                 "the project resumes from where you left off.", "info")

    def _open_project_dialog(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            title="Open project folder", initialdir=os.path.abspath(self._projects_dir()))
        if d:
            self._load_project(os.path.basename(d.rstrip("/")))

    def _refresh_project_menu(self):
        try:
            self.project_menu.configure(values=["Add new…"] + self._project_list())
            if self.project_name:
                self.project_menu.set(self.project_name)
        except Exception:
            pass

    def _stage_summary(self):
        if not any(self.stages.values()):
            return "beginning (nothing done yet)"
        done = [name for name, ok in self.stages.items() if ok]
        return "done: " + ", ".join(done)

    def _update_stage_chips(self):
        for key, label in (("transcribe", "Transcribe"), ("ai", "AI Plan"), ("cut", "Cut Video")):
            ok = self.stages.get(key, False)
            self.stage_chips[key].configure(
                text=("✓ " if ok else "○ ") + label,
                text_color=GREEN if ok else MUTED,
                fg_color=PANEL_2 if ok else PANEL,
                border_width=1, border_color="#1e5c3a" if ok else BORDER)

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

        self._card_project(left)
        self._card_transcription(left)
        self._card_ai(left)
        self._card_highlights(left)

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

    def _card_project(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(0, weight=1)
        self._section_title(card, "🗂", "Project", 0)

        proj_row = ctk.CTkFrame(card, fg_color="transparent")
        proj_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 0))
        proj_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(proj_row, text="Project", font=ctk.CTkFont(size=13), text_color=MUTED,
                     width=66, anchor="w").grid(row=0, column=0, sticky="w")
        self.project_menu = ctk.CTkOptionMenu(proj_row, values=["Add new…"] + self._project_list(),
                                              command=self._on_project_menu,
                                              fg_color=PANEL_2, button_color=ACCENT,
                                              corner_radius=8,
                                              font=ctk.CTkFont(size=13), height=36)
        self.project_menu.grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(proj_row, text="New", width=56, height=36, corner_radius=8,
                      font=ctk.CTkFont(size=13), fg_color=ACCENT, hover_color=ACCENT_D,
                      command=self._new_project).grid(row=0, column=2, padx=(8, 0))
        ctk.CTkButton(proj_row, text="Open…", width=62, height=36, corner_radius=8,
                      font=ctk.CTkFont(size=13), fg_color=PANEL_2, hover_color="#334155",
                      text_color=FG, border_width=1, border_color=BORDER,
                      command=self._open_project_dialog).grid(row=0, column=3, padx=(8, 0))

        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 0))
        self.stage_chips = {}
        for key, label in (("transcribe", "Transcribe"), ("ai", "AI Plan"), ("cut", "Cut Video")):
            c = ctk.CTkLabel(chips, text="○ " + label, font=ctk.CTkFont(size=12, weight="bold"),
                             text_color=MUTED, corner_radius=10, height=26, padx=12,
                             fg_color=PANEL, border_width=1, border_color=BORDER)
            c.pack(side="left", padx=(0, 8))
            self.stage_chips[key] = c
        self._update_stage_chips()

        self._file_row(card, "Video File", self.video_path, "video", 4)
        self._file_row(card, "Prompt", self.prompt_path, "prompt", 5)

    def _on_project_menu(self, name):
        if name == "Add new…" or not name:
            self._new_project()
            return
        self._load_project(name)

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

    def _card_highlights(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        card.grid_columnconfigure(0, weight=1)
        self._section_title(card, "🎬", "Highlights → Video", 0)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 0))
        self.clip_count_label = ctk.CTkLabel(head, text="no highlight.json loaded",
                                             font=ctk.CTkFont(size=12), text_color=MUTED)
        self.clip_count_label.pack(side="left")
        ctk.CTkButton(head, text="⟳ Reload", width=80, height=30, corner_radius=8,
                      font=ctk.CTkFont(size=12), fg_color=PANEL_2, hover_color="#334155",
                      text_color=FG, border_width=1, border_color=BORDER,
                      command=self._load_highlights).pack(side="right")

        self.clip_list = tk.Listbox(card, height=5, background="#0d1017", fg="#d7e2f5",
                                    selectbackground=ACCENT, selectforeground="#ffffff",
                                    highlightthickness=0, borderwidth=0, font=("Monospace", 10))
        self.clip_list.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 0))

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=3, column=0, sticky="ew", padx=16, pady=(10, 14))
        self.cut_btn = ctk.CTkButton(btns, text="✂  Cut Highlight Video", height=38, corner_radius=9,
                                     font=ctk.CTkFont(size=13, weight="bold"), fg_color=ACCENT,
                                     hover_color=ACCENT_D, state="disabled", command=self._cut_video)
        self.cut_btn.pack(side="left")
        self.open_btn = ctk.CTkButton(btns, text="Open Output", height=38, width=110, corner_radius=9,
                                      font=ctk.CTkFont(size=13), fg_color=PANEL_2, hover_color="#334155",
                                      text_color=FG, border_width=1, border_color=BORDER,
                                      state="disabled", command=self._open_output)
        self.open_btn.pack(side="left", padx=(8, 0))

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
        self.log_path = "app.log"
        try:
            open(self.log_path, "w", encoding="utf-8").close()
        except OSError:
            pass
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
        if self.project_name:
            self._save_project()

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
        dead = self._load_dead_models()
        ordered = [m for m in FREE_MODELS_FALLBACK if m not in dead]
        for m in models:
            if m not in ordered and m not in dead:
                ordered.append(m)
        current = self.ai_model.get()
        self.ai_model_combo.configure(values=ordered)
        self.ai_model.set(current if current in ordered else ordered[0])
        self._model_count = len(ordered)
        return True

    @staticmethod
    def _dead_models_file():
        return os.path.join(PROJECTS_DIR, ".dead_models.json")

    @classmethod
    def _load_dead_models(cls):
        try:
            with open(cls._dead_models_file(), "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (OSError, ValueError):
            return set()

    def _deal_model(self, model):
        if not hasattr(self, "_dead_models"):
            self._dead_models = set(self._load_dead_models())
        self._dead_models.add(model)
        try:
            os.makedirs(PROJECTS_DIR, exist_ok=True)
            with open(self._dead_models_file(), "w", encoding="utf-8") as f:
                json.dump(sorted(self._dead_models), f)
        except OSError:
            pass

        def gui_update():
            values = list(self.ai_model_combo.cget("values") or [])
            if model in values:
                values.remove(model)
                self.ai_model_combo.configure(values=values)
                if self.ai_model.get() == model:
                    self.ai_model.set(values[0] if values else model)
        try:
            self.root.after(0, gui_update)
        except RuntimeError:
            pass

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
        ts = time.strftime("%H:%M:%S")
        self.msg_queue.put((message, level, ts))
        try:
            with open("app.log", "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{level.upper():7}] {message}\n")
        except OSError:
            pass

    # ----------------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------------

    def _run(self):
        if self.running:
            return
        if not self.project_name:
            from tkinter import messagebox
            messagebox.showerror("No Project", "Create or open a project first —\n"
                                 "each movie gets its own project.")
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

        self._save_project()
        self.log("── Starting pipeline ──", "info")
        self.status_text.set("Running...")
        self.running = True
        self.run_btn.configure(text="■  Stop", fg_color="#b03a2e", hover_color="#8f2d24")
        self.status_dot.configure(text_color=RED)
        self.pb.set(0.05)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        try:
            if not self.stages.get("transcribe"):
                self._step_transcribe()
                self.stages["transcribe"] = True
                self._save_project()
                self.root.after(0, self._update_stage_chips)
            else:
                self.log("Transcribe: already done — skipping (resume).", "info")

            if self._stop_requested():
                raise StopRequested()

            if not self.stages.get("ai"):
                self._step_ai()
                self.stages["ai"] = True
                self._save_project()
                self.root.after(0, self._update_stage_chips)
            else:
                self.log("AI plan: already done — skipping (resume).", "info")
                self._load_highlights()

            if not self.stages.get("cut"):
                self.log("── Done ──", "success")
                self.log("Stages done: transcribe + AI plan. Cutting video…", "success")
                self.root.after(200, self._cut_video_after_pipeline)
            else:
                self.log("Cut: already done — project fully complete.", "success")
                self.log("── Done ──", "success")
        except StopRequested:
            self.log("Pipeline stopped by user.", "info")
        except Exception as e:
            self.log(f"Pipeline failed: {type(e).__name__}: {e}", "error")
        finally:
            self.root.after(0, self._reset_running)

    def _stop_requested(self):
        return False

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
        kwargs = {
            "language": language,
            "beam_size": 5,
            "task": "transcribe",
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 700,
                               "speech_pad_ms": 200},
            "condition_on_previous_text": False,
        }
        if not language:
            kwargs.pop("language")
        result = model.transcribe(self.video_path.get(), **kwargs)

        segments, info = result
        self.log(f"Detected language: {info.language}", "info")

        srt_path = self._project_file("video.srt")
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)
        count = 0
        dropped = 0
        # Open once, flush as we go: the SRT file survives even if a later
        # segment (or the AI step) fails.
        with open(srt_path, "w", encoding="utf-8") as f:
            try:
                for i, seg in enumerate(segments, 1):
                    text = seg.text.strip()
                    if not text:
                        continue
                    # Hallucination guard: music/silence segments have very
                    # low avg_logprob (real speech is usually > -1.6)
                    if getattr(seg, "no_speech_prob", 0.0) > 0.6 or \
                       getattr(seg, "avg_logprob", 0.0) < -2.0:
                        dropped += 1
                        continue
                    # Junk guard: no real letters (punctuation/emoji only)
                    letters = [ch for ch in text if ch.isalpha()]
                    if not letters:
                        dropped += 1
                        continue
                    start = format_timestamp(seg.start, always_include_hours=True, decimal_marker=",")
                    end = format_timestamp(seg.end, always_include_hours=True, decimal_marker=",")
                    f.write(f"{i}\n{start} --> {end}\n{text}\n")
                    f.flush()
                    count += 1
                    self.log(f"  [{start} → {end}] {text}", "info")
            except Exception as e:
                raise RuntimeError(f"Transcription failed at segment {count + 1}: {e}") from e

        if dropped:
            self.log(f"Skipped {dropped} music/noise segments (hallucination filter).", "info")
        self.log(f"SRT saved  →  {srt_path}  ({count} segments)", "success")
        self.pb.set(0.6)

    def _step_ai(self):
        srt_path = self._project_file("video.srt")
        with open(srt_path, "r", encoding="utf-8") as f:
            subtitle = f.read()
        with open(self.prompt_path.get(), "r", encoding="utf-8") as f:
            prompt = f.read()

        ai_model = self.ai_model.get().strip()
        self.log(f"AI →  OpenRouter  ·  {ai_model}", "info")

        text = self._call_openrouter_resilient(self.openrouter_key, ai_model, prompt, subtitle)
        try:
            data = self._parse_ai_response(text)
        except AiErrorResponse as e:
            self.log(f"AI refused: {e}", "error")
            self.log("Retrying with a tolerant instruction…", "info")
            text = self._call_openrouter_resilient(self.openrouter_key, ai_model,
                                                   prompt + TOLERANT_HINT, subtitle)
            data = self._parse_ai_response(text)

        with open(self._project_file("highlight.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        clips = data.get("clips", []) if isinstance(data, dict) else data
        self.log(f"highlight.json saved  →  ({len(clips)} clips)", "success")
        self.pb.set(1.0)

        self._load_highlights()
        self.root.after(200, self._cut_video_after_pipeline)

    @staticmethod
    def _parse_ai_response(text):
        if not text or not str(text).strip():
            raise AiErrorResponse("empty response from AI (content was null)")
        text = str(text).replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise AiErrorResponse(f"response was not valid JSON: {str(e)[:80]}") from e
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            raise AiErrorResponse(data["error"])
        if isinstance(data, dict) and "clips" in data:
            return data
        if isinstance(data, list):
            return {"clips": data}
        raise AiErrorResponse("unexpected response format")

    def _available_models(self):
        values = self.ai_model_combo.cget("values") or FREE_MODELS_FALLBACK
        return [v for v in values if v and v not in getattr(self, "_dead_models", ())]

    def _call_openrouter_resilient(self, api_key, first_model, prompt, subtitle):
        candidates = self._available_models()
        if first_model not in candidates and first_model not in getattr(self, "_dead_models", ()):
            candidates = [first_model] + candidates
        tried = []
        for model in candidates:
            if model in tried:
                continue
            tried.append(model)
            try:
                return self._call_openrouter(api_key, model, prompt, subtitle)
            except RateLimitedError:
                self.log(f"  {model} rate-limited (429) → trying next free model…", "error")
            except InvalidModelError:
                self.log(f"  {model} not available on OpenRouter anymore — removed from list.", "error")
                self._deal_model(model)
            except AiErrorResponse as e:
                self.log(f"  {model} bad reply ({e}) → trying next free model…", "error")
        live = [m for m in candidates if m not in getattr(self, "_dead_models", ())]
        if not live:
            raise InvalidModelError(
                "None of the configured models are available on OpenRouter anymore. "
                "Pick a model from the drop-down list on the right.")
        # last resort: give the very first live model one more shot
        self.log(f"  All {len(live)} models failed once — one final retry with {live[0]}…", "error")
        time.sleep(5)
        try:
            return self._call_openrouter(api_key, live[0], prompt, subtitle)
        except (RateLimitedError, AiErrorResponse) as e:
            raise RuntimeError(
                f"All free models failed (rate-limited or empty replies). Last error: {e}") from e

    @staticmethod
    def _retry_after_seconds(resp, fallback):
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return min(int(header), 30)
            except ValueError:
                pass
        return fallback

    @staticmethod
    def _call_openrouter(api_key, ai_model, prompt, subtitle):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": ai_model,
                   "messages": [{"role": "user", "content": f"{prompt}\n\nSubtitle:\n\n{subtitle}"}]}
        for attempt in range(3):
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
            if resp.status_code == 429:
                wait = App._retry_after_seconds(resp, fallback=2 * (attempt + 1))
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                if resp.status_code == 400 and "not a valid model ID" in resp.text:
                    raise InvalidModelError(ai_model)
                raise RuntimeError(
                    f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                raise AiErrorResponse(
                    f"empty content (finish_reason={choice.get('finish_reason', '?')})")
            return content
        raise RateLimitedError(ai_model)

    def _load_whisper(self):
        def job():
            self.log("Pre-loading Whisper model…", "info")
            try:
                WhisperModel(self.whisper_model.get(), device="cpu", compute_type="int8")
                self.log("Model ready — first Run will be fast now.", "success")
            except Exception as e:
                self.log(f"Model load failed: {e}", "error")
        threading.Thread(target=job, daemon=True).start()

    # ----------------------------------------------------------------
    # Highlights → video cutting
    # ----------------------------------------------------------------

    COLOR_PRESETS = {
        "original": "",
        "natural": "eq=contrast=1.05:saturation=1.05",
        "cinematic": "eq=contrast=1.12:saturation=1.08:brightness=0.01:gamma=0.98",
        "warm": "colorchannelmixer=rr=1.04:gg=1.0:bb=0.94",
        "cool": "colorchannelmixer=rr=0.95:gg=1.0:bb=1.05",
        "dramatic": "eq=contrast=1.18:saturation=1.0:brightness=-0.02",
        "desaturated": "eq=saturation=0.5",
        "high_contrast": "eq=contrast=1.25:saturation=1.1",
        "soft": "eq=contrast=0.95:saturation=0.94:brightness=0.02",
    }

    TRANSITION_DURATION = {"cut": 0.0, "fade": 0.4, "crossfade": 0.4,
                           "fade_black": 0.7, "fade_white": 0.7}

    _enc_cache = None

    def _load_highlights(self):
        path = self._project_file("highlight.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.highlights = json.load(f)
            clips = self.highlights.get("clips", [])
            self.clip_list.delete(0, "end")
            for c in clips:
                self.clip_list.insert("end", f"{c.get('start','?')} → {c.get('end','?')}   {c.get('reason','')}")
            self.clip_count_label.configure(
                text=f"{len(clips)} clips loaded" if clips else "0 clips in file")
            self.cut_btn.configure(state="normal" if clips else "disabled")
            self.log(f"Loaded {len(clips)} highlight clips from highlight.json", "success")
        except FileNotFoundError:
            self.highlights = []
            self.cut_btn.configure(state="disabled")
            self.clip_count_label.configure(text="no highlight.json yet")
        except Exception as e:
            self.log(f"Could not read highlight.json: {e}", "error")

    @staticmethod
    def _ts_to_seconds(ts):
        if isinstance(ts, (int, float)):
            return float(ts)
        parts = str(ts).replace(",", ".").split(":")
        secs = float(parts.pop())
        mult = 60
        while parts:
            secs += float(parts.pop()) * mult
            mult *= 60
        return secs

    @staticmethod
    def _has_audio(path):
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=60)
            return bool(out.stdout.strip())
        except Exception:
            return True

    def _cut_video(self):
        if self.cutting or self.running:
            return
        clips = (self.highlights or {}).get("clips", [])
        if not clips:
            from tkinter import messagebox
            messagebox.showerror("No highlights", "Load a highlight.json with clips first.")
            return
        video = self.video_path.get()
        if not os.path.exists(video):
            from tkinter import messagebox
            messagebox.showerror("Error", "Video not found:\n" + video)
            return

        self.cutting = True
        self.status_text.set("Cutting video…")
        self.cut_btn.configure(state="disabled", text="Cutting…")
        self.pb.set(0)
        threading.Thread(target=self._do_cut, args=(video, clips), daemon=True).start()

    def _do_cut(self, video, clips):
        try:
            os.makedirs(self._project_folder() or ".", exist_ok=True)
            w, h, fps, duration = self._probe_video(video)
            has_audio = self._has_audio(video)
            self.log(f"Source: {w}x{h} @ {fps}fps · {duration:.0f}s "
                     f"· audio={'yes' if has_audio else 'no'}", "info")

            jobs = []
            self.log(f"Editing plan → {len(clips)} clips", "info")
            for idx, clip in enumerate(clips, 1):
                s = self._ts_to_seconds(clip.get("start", 0))
                e = self._ts_to_seconds(clip.get("end", s))
                dur = e - s
                if dur < 0.3 or s >= max(duration - 0.1, 0):
                    self.log(f"  clip {idx} skipped (invalid range {s:.1f}→{e:.1f}s)", "error")
                    continue
                jobs.append((idx, clip, s, dur))
                self.log(f"  clip {idx}: {s:.1f}s → {e:.1f}s · "
                         f"fx={clip.get('visual_effect', 'none')} · "
                         f"color={clip.get('color', {}).get('preset', 'original')} · "
                         f"audio={clip.get('audio', 'original')}", "info")

            if not jobs:
                raise RuntimeError("No valid clips to cut.")

            enc = self._machine_encoder()
            cores = os.cpu_count() or 4
            workers = max(1, min(cores, len(jobs)) if enc["hw"] else max(1, cores // 2))
            threads = max(1, cores // max(workers, 1)) if not enc["hw"] else 2
            self.log(f"Engine: {enc['label']} · {workers} parallel workers · "
                     f"{threads} cpu threads each (total {cores} cores)", "info")

            tmp = os.path.join(self._project_folder() or ".", "cuts_tmp")
            os.makedirs(tmp, exist_ok=True)
            parts = [None] * len(jobs)
            done = [0]
            lock = threading.Lock()

            def run(job):
                idx, clip, s, dur = job
                out = os.path.join(tmp, f"clip_{idx:03d}.mp4")
                vf, af = self._build_clip_chain(idx, clip, w, h, fps, dur, has_audio)
                cmd = [FFMPEG, "-y", "-loglevel", "error"]
                if enc["hw"]:
                    cmd += ["-hwaccel", "vaapi", "-vaapi_device", enc["dev"]]
                cmd += ["-ss", f"{s:.3f}", "-t", f"{dur:.3f}", "-i", video]
                if enc["hw"]:
                    vf = vf + ",format=nv12,hwupload"
                cmd += ["-vf", vf]
                if af:
                    cmd += ["-af", af]
                cmd += ["-c:v"] + list(enc["codec"])
                if not enc["hw"]:
                    cmd += ["-threads", str(threads)]
                cmd += ["-c:a", "aac", "-b:a", "128k", "-y", out]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                if proc.returncode != 0 or not os.path.exists(out):
                    raise RuntimeError(
                        f"clip {idx} failed: {proc.stderr[-600:]}")
                with lock:
                    done[0] += 1
                self.root.after(0, lambda d=done[0]: self.pb.set(0.1 + 0.8 * d / len(jobs)))
                return out

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run, job): job[0] for job in jobs}
                for fut in as_completed(futures):
                    fut.result()

            parts = [os.path.join(tmp, f"clip_{seq:03d}.mp4") for seq, *_ in jobs]
            parts = [p for p in parts if os.path.exists(p)]
            listf = os.path.join(tmp, "list.txt")
            with open(listf, "w", encoding="utf-8") as f:
                for p in parts:
                    f.write(f"file '{os.path.abspath(p)}'\n")

            output = os.path.join(self._project_folder() or ".", "highlight_video.mp4")
            self.log(f"Concatenating {len(parts)} parts (stream copy, no re-encode)…", "info")
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                   "-i", listf, "-c", "copy", "-movflags", "+faststart", output]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if proc.returncode != 0 or not os.path.exists(output):
                raise RuntimeError(f"concat failed: {proc.stderr[-800:]}")
            shutil.rmtree(tmp, ignore_errors=True)

            self.pb.set(1.0)
            self.stages["cut"] = True
            self._save_project()
            self.root.after(0, self._update_stage_chips)
            self.root.after(0, lambda: self.open_btn.configure(state="normal"))
            self.log(f"✔ Highlight video saved → {os.path.relpath(output)}", "success")
        except Exception as e:
            self.log(f"Cutting failed: {e}", "error")
        finally:
            self.root.after(0, self._reset_cutting)

    @classmethod
    def _machine_encoder(cls):
        """Pick the best encoder for THIS machine (called once, cached)."""
        if cls._enc_cache is not None:
            return cls._enc_cache
        # Linux + Intel iGPU (VAAPI) → hardware encode, near-zero CPU load
        if sys.platform.startswith("linux"):
            for dev in sorted(glob("/dev/dri/renderD*")):
                probe = subprocess.run(
                    [FFMPEG, "-y", "-v", "error", "-hwaccel", "vaapi",
                     "-vaapi_device", dev, "-f", "lavfi",
                     "-i", "testsrc=duration=1:size=320x240:rate=30",
                     "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                     "-global_quality", "24", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=30)
                if probe.returncode == 0:
                    cls._enc_cache = {
                        "label": f"h264_vaapi hardware ({dev})", "hw": True,
                        "dev": dev, "codec": ("h264_vaapi", "-global_quality", "24")}
                    return cls._enc_cache
        cls._enc_cache = {"label": "libx264 cpu (thread-capped)", "hw": False,
                          "dev": None, "codec": ("libx264", "-preset", "veryfast", "-crf", "20")}
        return cls._enc_cache

    def _cut_video_after_pipeline(self):
        if self.running:
            self.root.after(500, self._cut_video_after_pipeline)
            return
        if self.stages.get("cut"):
            self.log("Cut already done — skipping.", "success")
            return
        self.log("Auto-starting video cut from edit plan…", "info")
        self._cut_video()

    def _reset_cutting(self):
        self.cutting = False
        self.status_text.set("Idle")
        self.cut_btn.configure(state="normal", text="✂  Cut Highlight Video")

    @staticmethod
    def _clamp_seconds(value, lo, hi):
        return min(max(value, lo), hi)

    def _build_clip_chain(self, idx, clip, w, h, fps, dur, has_audio):
        """Plain per-clip filter chain for -vf/-af (no labels, no trim —
        input -ss/-t handles the positioning)."""
        v = f"fps={fps},scale={w}:-2"

        # ---- visual effects ----
        fx = clip.get("visual_effect", "none") or "none"
        frames = max(int(dur * fps), 1)
        if fx == "slow_zoom_in":
            v += (f",zoompan=z='min(1+0.12*on/{frames},1.15)':d=1:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}")
        elif fx == "slow_zoom_out":
            v += (f",zoompan=z='max(1.12-0.12*on/{frames},1.0)':d=1:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}")
        elif fx == "soft_blur":
            v += ",boxblur=2:1"
        if fx == "subtle_vignette":
            v += ",vignette"

        # ---- color grading (preset + numeric params) ----
        color = clip.get("color") or {}
        preset = self.COLOR_PRESETS.get(color.get("preset", "original"), "")
        if preset:
            v += "," + preset
        nums = []
        for key, default in (("brightness", 0), ("contrast", 1),
                             ("saturation", 1), ("gamma", 1)):
            val = color.get(key)
            if isinstance(val, (int, float)) and val != default:
                nums.append(f"{key}={val}")
        if nums:
            v += ",eq=" + ":".join(nums)

        # ---- transitions (fade in / fade out) ----
        ti = clip.get("transition_in", "cut") or "cut"
        to = clip.get("transition_out", "cut") or "cut"
        fi = self.TRANSITION_DURATION.get(ti, 0.0)
        fo = self.TRANSITION_DURATION.get(to, 0.0)
        if fi:
            d = min(fi, dur / 3)
            v += f",fade=t=in:st=0:d={d:.3f}:color={'white' if ti == 'fade_white' else 'black'}"
        if fo:
            d = min(fo, dur / 3)
            v += (f",fade=t=out:st={max(dur - d, 0):.3f}:d={d:.3f}:"
                  f"color={'white' if to == 'fade_white' else 'black'}")

        # ---- audio treatment ----
        a = ""
        if has_audio:
            ap = clip.get("audio", "original") or "original"
            if ap in ("normalize", "normalize_and_fade"):
                a += ",loudnorm=I=-16:TP=-1.5:LRA=11"
            if ap in ("fade_in", "normalize_and_fade"):
                a += ",afade=t=in:st=0:d=0.3"
            if ap in ("fade_out", "normalize_and_fade"):
                a += f",afade=t=out:st={max(dur - 0.3, 0):.3f}:d=0.3"
            a = a.lstrip(",")

        return v, a

    def _probe_video(self, path):
        out = subprocess.run(
            [FFPROBE, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout)
        st = data["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        num, den = st["r_frame_rate"].split("/")
        fps = round(float(num) / (float(den) or 1.0)) or 25
        duration = float(data.get("format", {}).get("duration", 0))
        return w, h, fps, duration

    def _open_output(self):
        folder = self._project_folder() or "."
        path = os.path.abspath(os.path.join(folder, "highlight_video.mp4"))
        if os.path.exists(path):
            try:
                subprocess.Popen(["xdg-open", folder])
            except OSError:
                self.log(f"Open the folder manually: {folder}", "info")
        else:
            self.log("No output video yet — cut one first.", "info")


def main():
    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    main()