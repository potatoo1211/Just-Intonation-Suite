import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, font
import threading
import time
import numpy as np
from scipy.optimize import minimize
import sounddevice as sd
import mido
import pretty_midi
import soundfile as sf
from music21 import converter

# ====================================================
# === 翻訳辞書 (Translation Dictionary) ==============
# ====================================================

TRANSLATIONS = {
    # --- Tab Titles ---
    "tab_tuner": {"ja": " 🔧 チューニング作成 ", "en": " 🔧 Tuner Generator "},
    "tab_music": {"ja": " 🎼 ファイル解析＆書き出し ", "en": " 🎼 File Analyzer & Render "},
    "tab_synth": {"ja": " 🎹 リアルタイム・シンセ ", "en": " 🎹 Real-time Synth "},

    # --- Common / Menu ---
    "menu_lang": {"ja": "言語 (Language)", "en": "Language"},
    "err_title": {"ja": "エラー", "en": "Error"},
    "success_title": {"ja": "成功", "en": "Success"},
    "warn_title": {"ja": "警告", "en": "Warning"},

    # --- Tuner Tab ---
    "tuner_frame_cfg": {"ja": "設定", "en": "Configuration"},
    "tuner_lbl_base": {"ja": "基準音 (鍵盤で選択):", "en": "Base Notes (Select on Piano):"},
    "tuner_btn_calc": {"ja": "計算実行", "en": "Calculate"},
    "tuner_btn_save": {"ja": ".scl (Scala) で保存", "en": "Save .scl"},
    "tuner_frame_res": {"ja": "計算結果", "en": "Results"},
    "tuner_res_wait": {"ja": "計算中 (精度向上のため10回試行します)...\n", "en": "Calculating (running 10 trials for precision)...\n"},
    "tuner_res_copy": {"ja": "\n▼ コピペ用フォーマット (空白区切り):\n", "en": "\n▼ Copy-Paste Format (Space separated):\n"},
    "tuner_warn_calc": {"ja": "先に計算を実行してください。", "en": "Please calculate tuning first."},
    "tuner_success_save": {"ja": "ファイルを保存しました: ", "en": "File saved: "},
    "tuner_err_noselect": {"ja": "基準音が選択されていません。鍵盤をクリックして少なくとも1つ選択してください。", "en": "No base notes selected. Please select at least one note on the piano."},

    # --- Music Tab ---
    "music_frame_file": {"ja": "ファイル選択", "en": "File Selection"},
    "music_lbl_in": {"ja": "入力 MIDI:", "en": "Input MIDI:"},
    "music_btn_browse": {"ja": "参照...", "en": "Browse..."},
    "music_lbl_out": {"ja": "出力 WAV:", "en": "Output WAV:"},
    "music_btn_save": {"ja": "保存先...", "en": "Save As..."},
    "music_frame_opt": {"ja": "解析オプション", "en": "Analysis Options"},
    "music_lbl_mode": {"ja": "主音モード:", "en": "Tonic Mode:"},
    "music_rb_global": {"ja": "全体 (標準 - A=440Hz固定)", "en": "Global (Just Intonation - Fixed A=440Hz)"},
    "music_rb_center": {"ja": "中心周波数基準", "en": "Center Freq (Just Intonation)"},
    "music_rb_equal": {"ja": "平均律 (通常のMIDI再生)", "en": "Equal Temperament (Standard MIDI)"},
    "music_btn_run": {"ja": "解析 ＆ WAV書き出し実行", "en": "Run Analysis & Render WAV"},
    "music_frame_log": {"ja": "処理ログ", "en": "Process Log"},
    "music_err_nofile": {"ja": "MIDIファイルを選択してください。", "en": "Please select a MIDI file."},

    # --- Synth Tab ---
    "synth_frame_ctrl": {"ja": "コントロールパネル", "en": "Control Panel"},
    "synth_lbl_base": {"ja": "基準音 (鍵盤で選択):", "en": "Base Notes (Select on Piano):"},
    "synth_btn_opt": {"ja": "音律を最適化して適用", "en": "Optimize & Apply"},
    "synth_lbl_port": {"ja": "MIDI 入力ポート:", "en": "MIDI Input Port:"},
    "synth_btn_ref": {"ja": "更新", "en": "Refresh"},
    "synth_btn_start": {"ja": "▶ シンセ開始", "en": "▶ Start Synth"},
    "synth_btn_stop": {"ja": "■ 停止", "en": "■ Stop"},
    "synth_frame_log": {"ja": "システムログ", "en": "System Log"},
    "synth_err_port": {"ja": "MIDIポートが選択されていません", "en": "No MIDI port selected"},
    "synth_msg_start": {"ja": "シンセを開始しました。", "en": "Synth Started."},
    "synth_msg_stop": {"ja": "シンセを停止しました。", "en": "Synth Stopped."},
    "synth_err_input": {"ja": "基準音が選択されていません。", "en": "No base notes selected."},
    "synth_no_device": {"ja": "デバイスが見つかりません", "en": "No devices found"}
}

def get_text(key, lang):
    return TRANSLATIONS.get(key, {}).get(lang, key)

# ====================================================
# === 共通: 数学・最適化ロジック (Backend) ===========
# ====================================================

class OptimizationLogic:
    def __init__(self):
        self.R = np.array([1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8, 2])
        self.target_logs = np.log2(self.R)
        self.w = 0.0001
        self.nD = [1, self.w, 1, self.w, 1, 1, self.w, 1, self.w, 1, self.w, 1, 1]

    def build_A_from_d(self, d):
        A = np.empty(12, dtype=float)
        A[0] = 0.0
        A[1:] = np.cumsum(d)
        return A

    def score_for_A(self, A, Bs, tonic_usage_map=None):
        total = 0.0
        valid_keys = 0
        
        if not Bs: return 100.0

        for Bi in Bs:
            if tonic_usage_map is not None:
                if Bi not in tonic_usage_map: continue
                used_in_this_key = tonic_usage_map[Bi]
            else:
                used_in_this_key = None

            valid_keys += 1
            b = A[Bi]
            e = []
            for j in range(13):
                p = Bi + j
                d_val = A[p % 12] + p // 12 - b - self.target_logs[j]
                
                weight = self.nD[j]
                if tonic_usage_map is not None:
                    target_pc = (Bi + j) % 12
                    if (j < 12 and target_pc in used_in_this_key) or (j == 12):
                        weight = 1.0
                
                e.append(d_val * d_val * weight)
            total += np.mean(e)
            
        if valid_keys == 0 and tonic_usage_map is not None: return 100.0
        denom = valid_keys if tonic_usage_map is not None else len(Bs)
        return total / denom

    def objective(self, d, Bs, tonic_usage_map=None):
        A = self.build_A_from_d(d)
        return self.score_for_A(A, Bs, tonic_usage_map)

    def anneal(self, Bs, tonic_usage_map=None, trials=1500):
        rng = np.random.default_rng() 
        d = np.full(11, 1.0 / 12.0)
        best_d = d.copy()
        best_score = self.objective(d, Bs, tonic_usage_map)
        temp = 0.1
        decay = 0.9995

        for _ in range(trials):
            cand = d + rng.normal(0, 0.002, size=11)
            if np.any(cand <= 1e-6): continue
            if np.sum(cand) >= 1.0 - 1e-6: continue

            sc = self.objective(cand, Bs, tonic_usage_map)
            if sc < best_score or rng.random() < np.exp((best_score - sc) / temp):
                d, best_score = cand, sc
                best_d = d.copy()
            temp *= decay
        
        return best_d, best_score

    def run_optimization(self, Bs, tonic_usage_map=None, trials=1500):
        best_start_d = None
        best_start_score = 1e9

        for _ in range(10):
            d_cand, s_cand = self.anneal(Bs, tonic_usage_map, trials)
            if s_cand < best_start_score:
                best_start_score = s_cand
                best_start_d = d_cand.copy()

        bounds = [(1e-6, 1.0)] * 11
        cons = ({'type': 'ineq', 'fun': lambda x: 1.0 - 1e-6 - np.sum(x)},)
        
        res = minimize(lambda x: self.objective(x, Bs, tonic_usage_map),
                       x0=best_start_d, bounds=bounds, constraints=cons,
                       method='SLSQP',
                       options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        
        d_opt = res.x
        A_opt = self.build_A_from_d(d_opt)
        score = self.score_for_A(A_opt, Bs, tonic_usage_map)
        return A_opt, score

# ====================================================
# === GUI Custom Widgets / Helpers ===================
# ====================================================

class TextRedirector(object):
    """ 標準出力等をテキストウィジェットにリダイレクトするクラス """
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
    
    def flush(self):
        pass

class PianoSelector(tk.Canvas):
    """ 鍵盤を描画し、クリックで音を選択できるカスタムウィジェット """
    def __init__(self, parent, width=350, height=80, default_selection=None):
        super().__init__(parent, width=width, height=height, bg="#e0e0e0", highlightthickness=0)
        self.selected_notes = set(default_selection) if default_selection else set()
        
        # 鍵盤データ
        self.white_keys_idx = [0, 2, 4, 5, 7, 9, 11] # C, D, E, F, G, A, B
        self.black_keys_idx = [1, 3, 6, 8, 10]       # C#, D#, F#, G#, A#
        self.keys_info = []

        self.w_key_width = width / 7.0
        self.b_key_width = self.w_key_width * 0.6
        self.b_key_height = height * 0.6
        self.height = height

        self.draw_keys()
        self.bind("<Button-1>", self.on_click)

    def draw_keys(self):
        self.delete("all")
        self.keys_info = []

        # 白鍵の描画
        for i, note in enumerate(self.white_keys_idx):
            x1 = i * self.w_key_width
            y1 = 0
            x2 = x1 + self.w_key_width
            y2 = self.height
            
            fill_color = "#98fb98" if note in self.selected_notes else "white"
            outline = "#333333"
            
            tag = f"key_{note}"
            self.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=outline, tags=tag)
            
            # 音名ラベル
            name = self.get_note_name(note)
            self.create_text((x1+x2)/2, y2-15, text=name, fill="black", font=("Arial", 9, "bold"), tags=tag)
            
            self.keys_info.append({"note": note, "type": "white", "coords": (x1, y1, x2, y2)})

        # 黒鍵の描画
        black_positions_indices = [0, 1, 3, 4, 5]
        black_notes_map = {0:1, 1:3, 3:6, 4:8, 5:10}

        for w_idx in black_positions_indices:
            note = black_notes_map[w_idx]
            center_x = (w_idx + 1) * self.w_key_width
            x1 = center_x - (self.b_key_width / 2)
            y1 = 0
            x2 = center_x + (self.b_key_width / 2)
            y2 = self.b_key_height

            fill_color = "#228b22" if note in self.selected_notes else "black"
            
            tag = f"key_{note}"
            self.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#333333", tags=tag)
            
            name = self.get_note_name(note)
            self.create_text((x1+x2)/2, y2-12, text=name, fill="white", font=("Arial", 7), tags=tag)
            
            self.keys_info.append({"note": note, "type": "black", "coords": (x1, y1, x2, y2)})

    def get_note_name(self, note):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return names[note]

    def on_click(self, event):
        clicked_note = None
        
        # Check Black Keys first
        for k in self.keys_info:
            if k["type"] == "black":
                x1, y1, x2, y2 = k["coords"]
                if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                    clicked_note = k["note"]
                    break
        
        # Check White Keys
        if clicked_note is None:
            for k in self.keys_info:
                if k["type"] == "white":
                    x1, y1, x2, y2 = k["coords"]
                    if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                        clicked_note = k["note"]
                        break
        
        if clicked_note is not None:
            if clicked_note in self.selected_notes:
                self.selected_notes.remove(clicked_note)
            else:
                self.selected_notes.add(clicked_note)
            self.draw_keys()

    def get_selection(self):
        return sorted(list(self.selected_notes))

# ====================================================
# === Main App Class =================================
# ====================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.current_lang = "ja"  # Default Language
        
        self.title("Just Intonation Suite v1.7")
        self.geometry("980x780")
        
        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        default_font = ("Yu Gothic UI", 10)
        if "Meiryo UI" in font.families():
             default_font = ("Meiryo UI", 10)
        
        style.configure(".", font=default_font)
        style.configure("TButton", padding=6, font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("TLabelframe", font=(default_font[0], default_font[1], "bold"))
        style.configure("TLabelframe.Label", foreground="#333333")

        # --- Menu Bar ---
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        self.lang_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Language", menu=self.lang_menu)
        self.lang_menu.add_command(label="日本語 (Japanese)", command=lambda: self.switch_language("ja"))
        self.lang_menu.add_command(label="English", command=lambda: self.switch_language("en"))

        # --- Tabs ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=15)

        self.optimizer = OptimizationLogic()

        self.tab_tuner = TunerTab(self.notebook, self.optimizer)
        self.tab_music = MusicTab(self.notebook, self.optimizer)
        self.tab_synth = SynthTab(self.notebook, self.optimizer)

        self.notebook.add(self.tab_tuner, text="")
        self.notebook.add(self.tab_music, text="")
        self.notebook.add(self.tab_synth, text="")

        # Apply Initial Text
        self.update_ui_text()

    def switch_language(self, lang):
        self.current_lang = lang
        self.update_ui_text()

    def update_ui_text(self):
        lang = self.current_lang
        # Update Tab Titles
        self.notebook.tab(0, text=get_text("tab_tuner", lang))
        self.notebook.tab(1, text=get_text("tab_music", lang))
        self.notebook.tab(2, text=get_text("tab_synth", lang))
        
        self.tab_tuner.update_texts(lang)
        self.tab_music.update_texts(lang)
        self.tab_synth.update_texts(lang)

# ====================================================
# === Tab 1: Tuner Generator =========================
# ====================================================

class TunerTab(ttk.Frame):
    def __init__(self, parent, optimizer):
        super().__init__(parent)
        self.optimizer = optimizer
        self.parent_notebook = parent
        self.A_opt = None
        self.build_ui()

    def build_ui(self):
        main_layout = ttk.Frame(self)
        main_layout.pack(fill='both', expand=True, padx=15, pady=15)

        # Config Panel
        self.ctrl_frame = ttk.LabelFrame(main_layout, padding=15)
        self.ctrl_frame.pack(fill='x', pady=(0, 15))

        # Row 1: Label
        row1 = ttk.Frame(self.ctrl_frame)
        row1.pack(fill='x', pady=(0, 5))
        self.lbl_base = ttk.Label(row1, text="Base Notes:")
        self.lbl_base.pack(side='left')

        # Row 2: Piano & Buttons
        row2 = ttk.Frame(self.ctrl_frame)
        row2.pack(fill='x')

        # Piano Widget (Replaces Entry)
        self.piano = PianoSelector(row2, width=350, height=80, default_selection=[0, 4, 7])
        self.piano.pack(side='left', padx=(0, 15))

        # Buttons
        btn_frame = ttk.Frame(row2)
        btn_frame.pack(side='left', fill='y')
        
        self.btn_calc = ttk.Button(btn_frame, command=self.calculate)
        self.btn_calc.pack(fill='x', pady=2)
        
        self.btn_save = ttk.Button(btn_frame, command=self.save_scl)
        self.btn_save.pack(fill='x', pady=2)

        # Result Panel
        self.res_frame = ttk.LabelFrame(main_layout, padding=10)
        self.res_frame.pack(fill='both', expand=True)
        self.res_text = scrolledtext.ScrolledText(self.res_frame, height=15, font=("Consolas", 10))
        self.res_text.pack(fill='both', expand=True)

    def update_texts(self, lang):
        self.ctrl_frame.configure(text=get_text("tuner_frame_cfg", lang))
        self.lbl_base.configure(text=get_text("tuner_lbl_base", lang))
        self.btn_calc.configure(text=get_text("tuner_btn_calc", lang))
        self.btn_save.configure(text=get_text("tuner_btn_save", lang))
        self.res_frame.configure(text=get_text("tuner_frame_res", lang))

    def calculate(self):
        lang = self.winfo_toplevel().current_lang
        try:
            Bs = self.piano.get_selection()
            if not Bs:
                messagebox.showwarning(get_text("warn_title", lang), get_text("tuner_err_noselect", lang))
                return
            
            self.res_text.delete('1.0', 'end')
            self.res_text.insert('end', "Calculating (running 10 trials)...\n")
            self.update_idletasks()

            def task():
                A_opt, score = self.optimizer.run_optimization(Bs)
                self.A_opt = A_opt
                
                out = f"Optimization Score: {score:.9f}\n\n"
                out += "Index | Log2 Ratio  | Cents\n"
                out += "-"*35 + "\n"
                for i, a in enumerate(A_opt):
                    cents = (a - A_opt[0]) * 1200.0
                    out += f"{i:5d} | {a:.9f} | {cents:.2f}\n"
                
                formatted_list = " ".join(f"{x:.12f}" for x in A_opt)
                out += get_text("tuner_res_copy", "en") + formatted_list
                
                self.res_text.delete('1.0', 'end')
                self.res_text.insert('end', out)

            threading.Thread(target=task, daemon=True).start()

        except Exception as e:
            messagebox.showerror(get_text("err_title", lang), str(e))

    def save_scl(self):
        lang = self.winfo_toplevel().current_lang
        if self.A_opt is None: 
            messagebox.showwarning(get_text("warn_title", lang), get_text("tuner_warn_calc", lang))
            return
        
        filename = filedialog.asksaveasfilename(defaultextension=".scl", filetypes=[("Scala files", "*.scl")])
        if not filename: return
        try:
            cents = (self.A_opt - self.A_opt[0]) * 1200.0
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"! {filename}\n!\nGenerated via Just Intonation App\n")
                f.write("12\n")
                for c in cents[1:]: f.write(f"{c:.6f}\n")
                f.write("2/1\n")
            messagebox.showinfo(get_text("success_title", lang), get_text("tuner_success_save", lang) + filename)
        except Exception as e:
            messagebox.showerror(get_text("err_title", lang), str(e))

# ====================================================
# === Tab 2: File Analyzer ===========================
# ====================================================

class MusicTab(ttk.Frame):
    def __init__(self, parent, optimizer):
        super().__init__(parent)
        self.optimizer = optimizer
        self.midi_path = tk.StringVar()
        self.out_path = tk.StringVar(value="out_retuned.wav")
        self.tonic_mode = tk.StringVar(value="global")
        self.build_ui()

    def build_ui(self):
        main_layout = ttk.Frame(self)
        main_layout.pack(fill='both', expand=True, padx=15, pady=15)

        # File Selection
        self.frame_files = ttk.LabelFrame(main_layout, padding=15)
        self.frame_files.pack(fill='x', pady=(0, 15))

        f_grid = ttk.Frame(self.frame_files)
        f_grid.pack(fill='x')
        
        self.lbl_in = ttk.Label(f_grid)
        self.lbl_in.grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(f_grid, textvariable=self.midi_path, width=50).grid(row=0, column=1, padx=10, pady=5)
        self.btn_browse = ttk.Button(f_grid, command=self.browse_midi)
        self.btn_browse.grid(row=0, column=2, pady=5)
        
        self.lbl_out = ttk.Label(f_grid)
        self.lbl_out.grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(f_grid, textvariable=self.out_path, width=50).grid(row=1, column=1, padx=10, pady=5)
        self.btn_save_as = ttk.Button(f_grid, command=self.browse_out)
        self.btn_save_as.grid(row=1, column=2, pady=5)

        # Options
        self.frame_opts = ttk.LabelFrame(main_layout, padding=15)
        self.frame_opts.pack(fill='x', pady=(0, 15))
        
        self.lbl_mode = ttk.Label(self.frame_opts)
        self.lbl_mode.pack(side='left')
        
        self.rb_global = ttk.Radiobutton(self.frame_opts, variable=self.tonic_mode, value="global")
        self.rb_global.pack(side='left', padx=15)
        self.rb_center = ttk.Radiobutton(self.frame_opts, variable=self.tonic_mode, value="center")
        self.rb_center.pack(side='left', padx=15)
        self.rb_equal = ttk.Radiobutton(self.frame_opts, variable=self.tonic_mode, value="equal")
        self.rb_equal.pack(side='left', padx=15)

        # Run Button
        self.btn_run = ttk.Button(main_layout, command=self.start_process)
        self.btn_run.pack(pady=5, ipadx=10, ipady=5)

        # Log
        self.frame_log = ttk.LabelFrame(main_layout, padding=10)
        self.frame_log.pack(fill='both', expand=True, pady=(15, 0))
        self.log_widget = scrolledtext.ScrolledText(self.frame_log, height=10, state='disabled', font=("Consolas", 9))
        self.log_widget.pack(fill='both', expand=True)
        self.logger = TextRedirector(self.log_widget)

    def update_texts(self, lang):
        self.frame_files.configure(text=get_text("music_frame_file", lang))
        self.lbl_in.configure(text=get_text("music_lbl_in", lang))
        self.btn_browse.configure(text=get_text("music_btn_browse", lang))
        self.lbl_out.configure(text=get_text("music_lbl_out", lang))
        self.btn_save_as.configure(text=get_text("music_btn_save", lang))
        self.frame_opts.configure(text=get_text("music_frame_opt", lang))
        self.lbl_mode.configure(text=get_text("music_lbl_mode", lang))
        self.rb_global.configure(text=get_text("music_rb_global", lang))
        self.rb_center.configure(text=get_text("music_rb_center", lang))
        self.rb_equal.configure(text=get_text("music_rb_equal", lang))
        self.btn_run.configure(text=get_text("music_btn_run", lang))
        self.frame_log.configure(text=get_text("music_frame_log", lang))

    def log(self, msg): self.logger.write(msg + "\n")
    
    def browse_midi(self):
        p = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid *.midi")])
        if p: self.midi_path.set(p)
    
    def browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if p: self.out_path.set(p)

    def start_process(self):
        lang = self.winfo_toplevel().current_lang
        path = self.midi_path.get()
        if not path: 
            messagebox.showerror(get_text("err_title", lang), get_text("music_err_nofile", lang))
            return
        self.btn_run.configure(state='disabled')
        threading.Thread(target=self.process_thread, args=(path,), daemon=True).start()

    def process_thread(self, midi_path):
        try:
            mode = self.tonic_mode.get()
            self.log(f"--- Starting Analysis (Mode: {mode}) ---")
            
            if mode == 'equal':
                self.log("Equal Temperament selected. Skipping optimization.")
                A_opt = np.linspace(0, 1, 12, endpoint=False)
            else:
                Bs, tonic_usage_map = self.get_usage_by_key_signature(midi_path)
                if not Bs:
                    self.log("Key signature not found. Trying fallback analysis (music21)...")
                    Bs, tonic_usage_map = self.analyze_keys_fallback(midi_path)
                self.log(f"Detected Keys (Bs): {Bs}")
                self.log("Optimizing Tuning (Running trials for precision)...")
                A_opt, score = self.optimizer.run_optimization(Bs, tonic_usage_map, trials=2000)
                self.log(f"Optimization Score: {score:.8f}")
                self.log(f"Tuning (log2): {' '.join([f'{x:.4f}' for x in A_opt])}")

            self.log("Synthesizing Audio...")
            self.synthesize_midi_to_audio(midi_path, A_opt, self.out_path.get(), mode)
            
            self.log(f"Done! WAV saved to: {self.out_path.get()}")
            self.log("Automatic playback is disabled.")

        except Exception as e:
            self.log(f"Error: {e}")
            import traceback; self.log(traceback.format_exc())
        finally:
            self.btn_run.configure(state='normal')

    def get_usage_by_key_signature(self, midi_path):
        try: pm = pretty_midi.PrettyMIDI(midi_path)
        except: return [], {}
        ks_changes = pm.key_signature_changes
        if not ks_changes: return [], {}
        ks_changes.sort(key=lambda x: x.time)
        end_time = pm.get_end_time()
        segments = []
        for i, ks in enumerate(ks_changes):
            start = ks.time
            end = ks_changes[i+1].time if (i + 1 < len(ks_changes)) else end_time
            tonic = ks.key_number % 12
            segments.append((start, end, tonic))
        tonic_usage_map = {}
        found_tonics = set()
        for _, _, t in segments:
            found_tonics.add(t)
            if t not in tonic_usage_map: tonic_usage_map[t] = set()
        for inst in pm.instruments:
            if inst.is_drum: continue
            for note in inst.notes:
                for start, end, tonic in segments:
                    if start <= note.start < end:
                        tonic_usage_map[tonic].add(note.pitch % 12)
                        break
        return sorted(list(found_tonics)), tonic_usage_map

    def analyze_keys_fallback(self, midi_path):
        note_name_to_pc = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
        s = converter.parse(midi_path)
        parts = s.parts if s.parts else [s]
        usage_map = {}
        found_tonics = set()
        measure_window = 4
        for p in parts:
            measures = p.makeMeasures()
            total = len(measures.getElementsByClass('Measure'))
            i = 1
            while i <= total:
                j = min(i+measure_window-1, total)
                seg = measures.measures(i, j)
                current_tonic = None
                try:
                    k = seg.analyze('key')
                    if hasattr(k, 'tonic'):
                        tname = k.tonic.name
                        current_tonic = note_name_to_pc.get(tname, note_name_to_pc.get(tname[0], 0))
                except: pass
                if current_tonic is not None:
                    found_tonics.add(current_tonic)
                    if current_tonic not in usage_map: usage_map[current_tonic] = set()
                    flat_seg = seg.flatten()
                    for el in flat_seg.notes:
                        if el.isNote: usage_map[current_tonic].add(el.pitch.pitchClass)
                        elif el.isChord:
                            for n in el.pitches: usage_map[current_tonic].add(n.pitchClass)
                i += measure_window
        if not found_tonics: return [0], {0: set(range(12))}
        return sorted(list(found_tonics)), usage_map

    def synthesize_midi_to_audio(self, midi_path, A, out_wav, tonic_mode):
        sr = 44100
        pm = pretty_midi.PrettyMIDI(midi_path)
        note_counts = np.zeros(12, dtype=int)
        for inst in pm.instruments:
            if not inst.is_drum:
                for n in inst.notes: note_counts[n.pitch % 12] += 1
        most_freq_pc = int(np.argmax(note_counts))
        tonic_midi_center = 60 + most_freq_pc
        duration = pm.get_end_time()
        t = np.linspace(0, duration, int(sr*duration)+1)
        audio = np.zeros_like(t)

        def get_freq(note, A_opt):
            if tonic_mode == 'equal':
                return 440.0 * (2.0 ** ((note - 69) / 12.0))
            tonic_midi = tonic_midi_center if tonic_mode == 'center' else 60
            rel = note - tonic_midi
            octave = np.floor_divide(rel, 12)
            deg = int(rel % 12)
            tonic_freq = 440.0 * 2 ** ((tonic_midi - 69) / 12.0)
            return tonic_freq * (2.0 ** (A_opt[deg] + octave))

        for inst in pm.instruments:
            if inst.is_drum: continue
            for n in inst.notes:
                start_i = int(n.start * sr)
                end_i = int(n.end * sr)
                length = n.end - n.start
                if end_i <= start_i: continue
                freq = get_freq(n.pitch, A)
                cur_len = end_i - start_i
                tt = np.linspace(0, length, cur_len, endpoint=False)
                wave = np.sin(2*np.pi*freq*tt) + 0.5*np.sin(2*np.pi*2*freq*tt)
                att_t = 0.01; rel_t = 0.03
                env = np.ones_like(tt)
                att_s = int(att_t*sr); rel_s = int(rel_t*sr)
                if att_s < cur_len: env[:att_s] = np.linspace(0,1,att_s)
                if rel_s < cur_len: env[-rel_s:] = np.linspace(1,0,rel_s)
                amp = (n.velocity / 127.0) * 0.2
                wave *= env * amp
                target_slice = slice(start_i, start_i+len(wave))
                if audio[target_slice].shape[0] == wave.shape[0]: audio[target_slice] += wave
        maxv = np.max(np.abs(audio)) + 1e-12
        audio = audio / maxv * 0.95
        sf.write(out_wav, audio, sr)

# ====================================================
# === Tab 3: Real-time Synthesizer ===================
# ====================================================

class SynthTab(ttk.Frame):
    def __init__(self, parent, optimizer):
        super().__init__(parent)
        self.optimizer = optimizer
        self.running = False
        self.stream = None
        self.midi_thread = None
        self.active_notes = {}
        self.sustain_on = False
        self.sustain_notes = set()
        self.lock = threading.Lock()
        self.tuning_log2 = np.linspace(0, 1, 12, endpoint=False)
        
        self.build_ui()
        self.refresh_ports()

    def build_ui(self):
        main_layout = ttk.Frame(self)
        main_layout.pack(fill='both', expand=True, padx=15, pady=15)

        self.frame_ctrl = ttk.LabelFrame(main_layout, padding=15)
        self.frame_ctrl.pack(fill='x', pady=(0, 15))

        # Row 1: Piano & Optimize
        row1 = ttk.Frame(self.frame_ctrl)
        row1.pack(fill='x', pady=5)
        
        self.lbl_base = ttk.Label(row1, text="Base Notes:")
        self.lbl_base.pack(side='left', anchor='n', pady=5)
        
        self.piano = PianoSelector(row1, width=350, height=80, default_selection=[0, 7, 9])
        self.piano.pack(side='left', padx=10)
        
        self.btn_optimize = ttk.Button(row1, command=self.run_optimize)
        self.btn_optimize.pack(side='left', anchor='c')

        # Row 2: Port
        row2 = ttk.Frame(self.frame_ctrl)
        row2.pack(fill='x', pady=15)
        self.lbl_port = ttk.Label(row2)
        self.lbl_port.pack(side='left')
        self.combo_ports = ttk.Combobox(row2, state="readonly", width=23)
        self.combo_ports.pack(side='left', padx=10)
        self.btn_refresh = ttk.Button(row2, command=self.refresh_ports, width=6)
        self.btn_refresh.pack(side='left')

        # Row 3: Start/Stop
        row3 = ttk.Frame(self.frame_ctrl)
        row3.pack(fill='x', pady=(5, 5))
        self.btn_start = ttk.Button(row3, command=self.start_synth)
        self.btn_start.pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.btn_stop = ttk.Button(row3, command=self.stop_synth, state='disabled')
        self.btn_stop.pack(side='left', expand=True, fill='x', padx=(5, 0))

        # Log
        self.frame_log = ttk.LabelFrame(main_layout, padding=10)
        self.frame_log.pack(fill='both', expand=True)
        
        self.log_widget = scrolledtext.ScrolledText(self.frame_log, height=10, state='disabled', font=("Consolas", 9))
        self.log_widget.pack(fill='both', expand=True)
        self.logger = TextRedirector(self.log_widget)

    def update_texts(self, lang):
        self.frame_ctrl.configure(text=get_text("synth_frame_ctrl", lang))
        self.lbl_base.configure(text=get_text("synth_lbl_base", lang))
        self.btn_optimize.configure(text=get_text("synth_btn_opt", lang))
        self.lbl_port.configure(text=get_text("synth_lbl_port", lang))
        self.btn_refresh.configure(text=get_text("synth_btn_ref", lang))
        self.btn_start.configure(text=get_text("synth_btn_start", lang))
        self.btn_stop.configure(text=get_text("synth_btn_stop", lang))
        self.frame_log.configure(text=get_text("synth_frame_log", lang))

    def log(self, message):
        self.logger.write(message + "\n")

    def refresh_ports(self):
        ports = mido.get_input_names()
        self.combo_ports['values'] = ports
        lang = self.winfo_toplevel().current_lang
        if ports: self.combo_ports.current(0)
        else: self.combo_ports.set(get_text("synth_no_device", lang))

    def run_optimize(self):
        lang = self.winfo_toplevel().current_lang
        try:
            Bs = self.piano.get_selection()
            if not Bs:
                messagebox.showerror(get_text("err_title", lang), get_text("synth_err_input", lang))
                return

            self.log(f"Optimizing for Base Notes: {Bs} (Running 10 trials)...")
            
            def task():
                A_opt, score = self.optimizer.run_optimization(Bs)
                self.tuning_log2 = A_opt
                self.log(f"Optimization Complete. Score: {score:.8f}")
                self.log(f"Tuning: {' '.join([f'{x:.4f}' for x in A_opt])}")

            threading.Thread(target=task, daemon=True).start()
        except Exception as e:
            self.log(f"Error: {e}")

    def audio_callback(self, outdata, frames, time_info, status):
        SAMPLE_RATE = 44100
        ATTACK, DECAY, SUSTAIN_LVL, RELEASE = 0.02, 0.1, 0.7, 0.3
        t = np.arange(frames) / SAMPLE_RATE
        signal = np.zeros(frames, dtype=np.float32)

        with self.lock:
            to_delete = []
            for note, d in self.active_notes.items():
                dt = frames / SAMPLE_RATE
                d["time"] += dt
                f = d["freq"]
                wave = (
                    np.sin(2 * np.pi * f * (t + d["phase"])) * 0.6 +
                    np.sin(2 * np.pi * f * 2 * (t + d["phase"])) * 0.3 +
                    np.sin(2 * np.pi * f * 3 * (t + d["phase"])) * 0.1
                )
                if not d["release"]:
                    if d["time"] < ATTACK: env = d["time"] / ATTACK
                    elif d["time"] < ATTACK + DECAY: env = 1 - (1 - SUSTAIN_LVL) * ((d["time"] - ATTACK) / DECAY)
                    else: env = SUSTAIN_LVL
                else:
                    d["release_time"] += dt
                    env = d["release_start"] * (1 - d["release_time"] / RELEASE)
                    if env <= 0:
                        to_delete.append(note)
                        continue

                d["env"] = env
                d["phase"] += dt
                signal += wave * env * d["amp"]

            for note in to_delete: self.active_notes.pop(note, None)

        outdata[:] = np.clip(signal, -1, 1).reshape(-1, 1)

    def midi_listener_task(self, portname):
        BASE_FREQ = 261.63
        try:
            with mido.open_input(portname) as inport:
                self.log(f"Listening on {portname}")
                while self.running:
                    for msg in inport.iter_pending():
                        if hasattr(msg, "channel") and msg.channel != 0: continue
                        if msg.type == "note_on" and msg.velocity > 0:
                            note = msg.note
                            base = note % 12
                            octave = note // 12 - 5
                            freq = BASE_FREQ * (2 ** self.tuning_log2[base]) * (2 ** octave)
                            amp = msg.velocity / 127.0 * 0.4
                            with self.lock:
                                self.active_notes[note] = {
                                    "phase": 0.0, "freq": freq, "amp": amp,
                                    "time": 0.0, "env": 0.0,
                                    "release": False, "release_time": 0.0, "release_start": 0.0,
                                }
                        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                            note = msg.note
                            with self.lock:
                                if self.sustain_on: self.sustain_notes.add(note)
                                elif note in self.active_notes:
                                    d = self.active_notes[note]
                                    d["release"] = True; d["release_time"] = 0.0; d["release_start"] = d["env"]
                        elif msg.type == "control_change" and msg.control == 64:
                            if msg.value >= 64: self.sustain_on = True
                            else:
                                self.sustain_on = False
                                with self.lock:
                                    for n in list(self.sustain_notes):
                                        if n in self.active_notes:
                                            d = self.active_notes[n]
                                            d["release"] = True; d["release_time"] = 0.0; d["release_start"] = d["env"]
                                    self.sustain_notes.clear()
                    time.sleep(0.001)
        except Exception as e:
            self.log(f"MIDI Error: {e}")

    def start_synth(self):
        port = self.combo_ports.get()
        lang = self.winfo_toplevel().current_lang
        if not port or port == get_text("synth_no_device", lang):
            messagebox.showerror(get_text("err_title", lang), get_text("synth_err_port", lang))
            return
        self.running = True
        self.active_notes = {}
        try:
            self.stream = sd.OutputStream(samplerate=44100, channels=1, callback=self.audio_callback, blocksize=256, latency="low")
            self.stream.start()
        except Exception as e:
            messagebox.showerror(get_text("err_title", lang), str(e))
            self.running = False
            return
        self.midi_thread = threading.Thread(target=self.midi_listener_task, args=(port,), daemon=True)
        self.midi_thread.start()
        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.log("Synth Started.")

    def stop_synth(self):
        self.running = False
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self.log("Synth Stopped.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
