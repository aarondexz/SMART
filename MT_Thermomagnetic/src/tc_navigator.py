# %%
# =======================================================
# Multi-sample Tc Navigator - Standalone App (Tkinter)
# Publication build: Arial 12, subscripted phase labels,
# in-plot Tc annotations, half-page (6.5 x 4.5 in) figures.
# =======================================================

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path
import json
import re
import datetime
import os

# --- Publication Font Configuration (npj / Nature Portfolio) ---
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['pdf.fonttype'] = 42   # editable text in Illustrator
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # SVG keeps live text, not outlines

# --- Theme Configuration ---
THEME_AVAILABLE = False
try:
    import ttkbootstrap as tb
    THEME_AVAILABLE = True
except ImportError:
    tb = tk

def create_button(p, t, c, s=None, **kw):
    if THEME_AVAILABLE and s:
        return tb.Button(p, text=t, command=c, bootstyle=s, **kw)
    return ttk.Button(p, text=t, command=c, **kw)

# --- Utility Functions ---

PHASE_COLOR_MAP = {
    "Sm1Fe12": "#00FFFF",    
    "Sm3Fe29": "#FF50FF",    
    "Sm2Fe17": "darkorange",
    "Sm1Fe2":  "darkviolet"  
}

def get_phase_color(phase_name, default_color):
    return PHASE_COLOR_MAP.get(phase_name, default_color)

def format_phase_label(name):
    """Sm2Fe17 -> Sm₂Fe₁₇; a stoichiometric 1 is omitted (Sm1Fe12 -> SmFe₁₂)."""
    if not isinstance(name, str) or not name.strip():
        return str(name)
    tokens = re.findall(r'(\d+\.?\d*|[^\d]+)', name)
    out = ""
    for t in tokens:
        if t[0].isdigit():
            try:
                if float(t) == 1.0:      # skip "1", "1.0", "1."
                    continue
            except ValueError:
                pass
            out += "_{" + t + "}"
        else:
            out += t.replace(" ", r"\ ")
    return r"$\mathregular{" + out + "}$"

def tidy_axis(ax, fs, sci=False):
    """Uniform tick styling; optional scientific offset so derivative tick
    labels stay short (they are what pushes the y-labels out of alignment)."""
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    if sci:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(fs * 0.9)


def annotate_tc(ax, T, M, tc, m_tc, text, color, fs):
    """
    Place a Tc label in the empty space beside the transition instead of on top
    of the curve. Picks the side the curve is falling away from and draws a thin
    leader line back to the marker.
    """
    side = 1
    try:
        span = max((float(np.max(T)) - float(np.min(T))) * 0.05, 1.0)
        left = M[(T >= tc - span) & (T < tc)]
        right = M[(T > tc) & (T <= tc + span)]
        if len(left) > 0 and len(right) > 0:
            # falling curve -> free space is up-right; rising -> up-left
            side = 1 if np.mean(right) < np.mean(left) else -1

        # keep the label inside the axes if the transition sits near an edge
        t0, t1 = float(np.min(T)), float(np.max(T))
        if t1 > t0:
            pos = (tc - t0) / (t1 - t0)
            if pos > 0.85: side = -1
            elif pos < 0.15: side = 1
    except Exception:
        side = 1

    ax.annotate(
        text,
        xy=(tc, m_tc),
        xytext=(13 * side, 17), textcoords='offset points',
        color=color, fontsize=fs, linespacing=1.35,
        ha='left' if side > 0 else 'right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),
        arrowprops=dict(arrowstyle='-', color=color, lw=0.8, shrinkA=2, shrinkB=5),
        annotation_clip=False,
    )


def add_label_headroom(ax, frac=0.12):
    """Expand the top of the y-axis so annotations are not clipped."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * frac)


def safe_isnan(val):
    """Safely check for NaN/None without throwing Numpy type errors"""
    return val is None or pd.isna(val)

def extract_data(filename: Path) -> pd.DataFrame:
    start_row = 0
    with open(filename, 'r', errors='ignore') as f:
        for i, line in enumerate(f):
            if "Temperature (K)" in line:
                start_row = i
                break
    
    try:
        df = pd.read_csv(filename, skiprows=start_row, delimiter=",")
    except Exception:
        df = pd.read_csv(filename, skiprows=start_row, delimiter=r"\s+")

    possible_T = [c for c in df.columns if "Temperature" in c and "(K)" in c]
    possible_M = [c for c in df.columns if "Moment" in c and "(emu)" in c]
    
    if not possible_T or not possible_M:
        possible_T = [c for c in df.columns if "Temp" in c]
        possible_M = [c for c in df.columns if "Moment" in c]
        if not possible_T or not possible_M:
            raise ValueError(f"Missing Temperature/Moment columns in {filename}")

    temp_col, moment_col = possible_T[0], possible_M[0]
    df = df[[temp_col, moment_col]].copy()
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[moment_col] = pd.to_numeric(df[moment_col], errors='coerce')
    df = df.dropna()
    df = df.loc[df[temp_col].diff() != 0].reset_index(drop=True)
    df = df.rename(columns={temp_col: "Temperature (K)", moment_col: "Moment (emu)"})
    return df

def line(x, a, b): 
    return a * x + b

def ensure_odd_smooth(smooth: int, n: int) -> int:
    if smooth < 5: smooth = 5
    if smooth >= n: smooth = n - 1
    if smooth % 2 == 0: smooth += 1
    return max(3, smooth)

def get_derivative(T, M, smooth_window=51):
    n = len(T)
    if n < 5: return np.zeros_like(M)
    smooth_window = ensure_odd_smooth(int(smooth_window), n)
    try:
        M_smooth = savgol_filter(M, smooth_window, 3)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dM_dT = np.gradient(M_smooth, T)
            dM_dT = np.nan_to_num(dM_dT, nan=0.0, posinf=0.0, neginf=0.0)
            
            edge = max(5, smooth_window // 2)
            if len(dM_dT) > 2 * edge:
                dM_dT[:edge] = 0.0
                dM_dT[-edge:] = 0.0
    except:
        dM_dT = np.zeros_like(M)
    return dM_dT

def fit_two_ranges_logic(T, M, r1, r2):
    mask1 = (T >= r1[0]) & (T <= r1[1])
    mask2 = (T >= r2[0]) & (T <= r2[1])
    
    T1, M1 = T[mask1], M[mask1]
    T2, M2 = T[mask2], M[mask2]
    
    if len(T1) < 3 or len(T2) < 3: 
        return np.nan, None, None, None
        
    try:
        p1, _ = curve_fit(line, T1, M1)
        p2, _ = curve_fit(line, T2, M2)
    except RuntimeError:
        return np.nan, None, None, None
        
    a1, b1 = p1
    a2, b2 = p2
    
    if np.isclose(a1, a2): 
        return np.nan, p1, p2, None
        
    Tc = (b2 - b1) / (a1 - a2)
    M_at_Tc = a1 * Tc + b1
    return float(Tc), p1, p2, float(M_at_Tc)

def parse_formula(formula):
    if not isinstance(formula, str): return pd.Series()
    matches = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
    data = {}
    for el, num in matches:
        data[el] = float(num) if num else 1.0
    return pd.Series(data)

# --- UI Helpers ---

class DualSlider(ttk.LabelFrame):
    def __init__(self, master, label, from_=0, to=100, initial=(20, 80), callback=None):
        kwargs = {"text": label, "padding": 5}
        if THEME_AVAILABLE: kwargs["bootstyle"] = "info"
        super().__init__(master, **kwargs)
        
        self.callback = callback
        self.var_min = tk.DoubleVar(value=initial[0])
        self.var_max = tk.DoubleVar(value=initial[1])
        self.min_limit = from_
        self.max_limit = to
        
        self.columnconfigure(1, weight=1)
        
        ttk.Label(self, text="Min:").grid(row=0, column=0, padx=2)
        self.scale_min = ttk.Scale(self, from_=from_, to=to, variable=self.var_min, command=self._on_min_scale)
        self.scale_min.grid(row=0, column=1, sticky="ew", padx=2)
        self.entry_min = ttk.Entry(self, textvariable=self.var_min, width=6)
        self.entry_min.grid(row=0, column=2, padx=2)
        self.entry_min.bind('<Return>', self._on_min_entry)
        self.entry_min.bind('<FocusOut>', self._on_min_entry)
        
        ttk.Label(self, text="Max:").grid(row=1, column=0, padx=2)
        self.scale_max = ttk.Scale(self, from_=from_, to=to, variable=self.var_max, command=self._on_max_scale)
        self.scale_max.grid(row=1, column=1, sticky="ew", padx=2)
        self.entry_max = ttk.Entry(self, textvariable=self.var_max, width=6)
        self.entry_max.grid(row=1, column=2, padx=2)
        self.entry_max.bind('<Return>', self._on_max_entry)
        self.entry_max.bind('<FocusOut>', self._on_max_entry)

    def _on_min_scale(self, val):
        val = float(val)
        if val > self.var_max.get():
            self.var_max.set(val)
        if self.callback: self.callback()

    def _on_max_scale(self, val):
        val = float(val)
        if val < self.var_min.get():
            self.var_min.set(val)
        if self.callback: self.callback()

    def _on_min_entry(self, event):
        try:
            val = float(self.var_min.get())
            val = max(self.min_limit, min(self.max_limit, val))
            self.var_min.set(val)
            if val > self.var_max.get(): self.var_max.set(val)
            if self.callback: self.callback()
        except ValueError: pass

    def _on_max_entry(self, event):
        try:
            val = float(self.var_max.get())
            val = max(self.min_limit, min(self.max_limit, val))
            self.var_max.set(val)
            if val < self.var_min.get(): self.var_min.set(val)
            if self.callback: self.callback()
        except ValueError: pass

    def set_range(self, start, end):
        start = max(self.min_limit, min(self.max_limit, start))
        end = max(self.min_limit, min(self.max_limit, end))
        self.var_min.set(start)
        self.var_max.set(end)

    def get_range(self):
        return (self.var_min.get(), self.var_max.get())

    def update_limits(self, min_val, max_val):
        self.min_limit = min_val
        self.max_limit = max_val
        self.scale_min.config(from_=min_val, to=max_val)
        self.scale_max.config(from_=min_val, to=max_val)

# --- Main Application ---

class TcNavigatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tc Navigator Pro - Desktop App")
        self.root.geometry("1400x950")
        
        # State Variables
        self.folder = None
        self.files = []
        self.idx = 0
        self.cache = {}
        self.analysis_state = {}
        self.phase_dict = {}
        self.programmatic_update = False
        self.df_view = pd.DataFrame()

        self._setup_ui()

    def _setup_ui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_left = ttk.Frame(self.paned, padding=10, width=420)
        self.paned.add(self.frame_left, weight=1)

        self.frame_right = ttk.Frame(self.paned)
        self.paned.add(self.frame_right, weight=3)

        # 0. Project & Data Management (Professional Flow)
        lf_file = ttk.LabelFrame(self.frame_left, text="0. Project & Data", padding=5)
        lf_file.pack(fill="x", pady=(0, 5))

        btn_row_1 = ttk.Frame(lf_file)
        btn_row_1.pack(fill="x", pady=2)
        create_button(btn_row_1, "📂 Start New (Load Folder)", self._ui_load_folder, s="primary").pack(side="left", expand=True, fill="x", padx=1)
        create_button(btn_row_1, "📜 Load Phases (JSON)", self._ui_load_phases, s="primary").pack(side="left", expand=True, fill="x", padx=1)
        
        btn_row_2 = ttk.Frame(lf_file)
        btn_row_2.pack(fill="x", pady=2)
        create_button(btn_row_2, "💾 Save Project", self._save_session, s="success").pack(side="left", expand=True, fill="x", padx=1)
        create_button(btn_row_2, "📂 Load Project", self._load_session, s="success").pack(side="left", expand=True, fill="x", padx=1)

        # 1. Navigation
        lf_nav = ttk.LabelFrame(self.frame_left, text="1. Navigation", padding=5)
        if THEME_AVAILABLE: lf_nav.configure(bootstyle="primary")
        lf_nav.pack(fill="x", pady=5)
        
        self.var_sample_name = tk.StringVar()
        self.cb_samples = ttk.Combobox(lf_nav, textvariable=self.var_sample_name, state="readonly")
        self.cb_samples.pack(fill="x", pady=2)
        self.cb_samples.bind("<<ComboboxSelected>>", self._on_sample_select)
        
        btn_frame = ttk.Frame(lf_nav)
        btn_frame.pack(fill="x", pady=2)
        create_button(btn_frame, "◀ Prev", self._prev_sample, s="primary-outline").pack(side="left", expand=True, fill="x", padx=1)
        create_button(btn_frame, "Next ▶", self._next_sample, s="primary-outline").pack(side="right", expand=True, fill="x", padx=1)

        # 2. Settings
        lf_settings = ttk.LabelFrame(self.frame_left, text="2. View Settings", padding=5)
        lf_settings.pack(fill="x", pady=5)
        
        self.var_curve = tk.StringVar(value="Heating")
        
        if THEME_AVAILABLE:
            tb.Radiobutton(lf_settings, text="Heating", variable=self.var_curve, value="Heating", command=self._refresh_view, bootstyle="info").pack(anchor="w")
            tb.Radiobutton(lf_settings, text="Cooling", variable=self.var_curve, value="Cooling", command=self._refresh_view, bootstyle="info").pack(anchor="w")
        else:
            ttk.Radiobutton(lf_settings, text="Heating", variable=self.var_curve, value="Heating", command=self._refresh_view).pack(anchor="w")
            ttk.Radiobutton(lf_settings, text="Cooling", variable=self.var_curve, value="Cooling", command=self._refresh_view).pack(anchor="w")
        
        ttk.Label(lf_settings, text="Smooth Window:").pack(anchor="w", pady=(5, 0))
        self.var_smooth = tk.IntVar(value=51)
        sc_smooth = ttk.Scale(lf_settings, from_=5, to=201, variable=self.var_smooth, command=lambda v: self._update_plot())
        if THEME_AVAILABLE: sc_smooth.configure(bootstyle="success")
        sc_smooth.pack(fill="x")

        self.slider_view = DualSlider(lf_settings, "View Temp Range", callback=self._update_plot)
        self.slider_view.pack(fill="x", pady=5)

        # 3. Phase Management
        lf_phases = ttk.LabelFrame(self.frame_left, text="3. Phases", padding=5)
        lf_phases.pack(fill="x", pady=5, expand=True)
        
        self.lb_phases = tk.Listbox(lf_phases, height=6, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=1)
        self.lb_phases.pack(fill="x", pady=2)
        self.lb_phases.bind("<<ListboxSelect>>", self._on_phase_select)
        
        ttk.Label(lf_phases, text="Phase Name:").pack(anchor="w")
        self.var_phase_name = tk.StringVar()
        self.cb_phase_name = ttk.Combobox(lf_phases, textvariable=self.var_phase_name)
        self.cb_phase_name.pack(fill="x", pady=2)
        self.cb_phase_name.bind("<<ComboboxSelected>>", self._on_name_change)
        self.cb_phase_name.bind("<KeyRelease>", self._on_name_change)
        
        btn_phase_frame = ttk.Frame(lf_phases)
        btn_phase_frame.pack(fill="x", pady=2)
        create_button(btn_phase_frame, "Add New", self._add_phase, s="success").pack(side="left", expand=True, fill="x", padx=1)
        create_button(btn_phase_frame, "Delete", self._del_phase, s="danger").pack(side="left", expand=True, fill="x", padx=1)
        create_button(btn_phase_frame, "Auto (JSON)", self._auto_detect, s="info").pack(side="left", expand=True, fill="x", padx=1)

        # 4. Fitting
        lf_fit = ttk.LabelFrame(self.frame_left, text="4. Fit Ranges (Active)", padding=5)
        lf_fit.pack(fill="x", pady=5)
        
        self.slider_r1 = DualSlider(lf_fit, "Range 1 (Tangent)", callback=self._on_fit_change)
        self.slider_r1.pack(fill="x", pady=2)
        
        self.slider_r2 = DualSlider(lf_fit, "Range 2 (Baseline)", callback=self._on_fit_change)
        self.slider_r2.pack(fill="x", pady=2)

        # 5. Storage / Export
        lf_store = ttk.LabelFrame(self.frame_left, text="5. Final Export", padding=5)
        lf_store.pack(fill="x", pady=5, side="bottom")
        
        exprt_frame = ttk.Frame(lf_store)
        exprt_frame.pack(fill="x", pady=5)
        
        # Half a Word page (Letter, 1" margins) = 6.5 x 4.5 inches
        ttk.Label(exprt_frame, text="Fig W:").grid(row=0, column=0, padx=2)
        self.var_ext_w = tk.DoubleVar(value=6.5)
        entry_w = ttk.Entry(exprt_frame, textvariable=self.var_ext_w, width=5)
        entry_w.grid(row=0, column=1)
        entry_w.bind('<Return>', self._apply_live_settings)
        entry_w.bind('<FocusOut>', self._apply_live_settings)
        
        ttk.Label(exprt_frame, text="Fig H:").grid(row=0, column=2, padx=2)
        self.var_ext_h = tk.DoubleVar(value=4.5)
        entry_h = ttk.Entry(exprt_frame, textvariable=self.var_ext_h, width=5)
        entry_h.grid(row=0, column=3)
        entry_h.bind('<Return>', self._apply_live_settings)
        entry_h.bind('<FocusOut>', self._apply_live_settings)
        
        ttk.Label(exprt_frame, text="Font:").grid(row=0, column=4, padx=2)
        self.var_ext_font = tk.DoubleVar(value=14.0)
        entry_f = ttk.Entry(exprt_frame, textvariable=self.var_ext_font, width=5)
        entry_f.grid(row=0, column=5)
        entry_f.bind('<Return>', self._apply_live_settings)
        entry_f.bind('<FocusOut>', self._apply_live_settings)

        fmt_frame = ttk.Frame(lf_store)
        fmt_frame.pack(fill="x", pady=2)
        ttk.Label(fmt_frame, text="Format:").pack(side="left", padx=(0, 4))
        self.var_ext_fmt = tk.StringVar(value="SVG + PNG")
        ttk.Combobox(fmt_frame, textvariable=self.var_ext_fmt, state="readonly",
                     values=["SVG", "PDF", "SVG + PNG", "PNG"], width=12).pack(side="left")

        # Optional: export only the M(T) panel (recommended for a half-page figure)
        self.var_single_panel = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_store, text="Export M(T) panel only (no derivatives)",
                        variable=self.var_single_panel).pack(anchor="w", pady=2)

        create_button(lf_store, "Export Results (CSV + PNGs)", self._export_project, s="warning").pack(fill="x", pady=2)
        create_button(lf_store, "Merge SEM Data", self._merge_sem_data, s="info").pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(self.frame_left, text="Please load a folder or a project to begin.", foreground="#44ccff" if THEME_AVAILABLE else "blue", wraplength=380)
        self.lbl_status.pack(side="bottom", pady=5)

        # --- RIGHT PANEL (MATPLOTLIB) ---
        self.frame_toolbar = ttk.Frame(self.frame_right)
        self.frame_toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.frame_plot = tk.Frame(self.frame_right, bg="#2b2b2b") 
        self.frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6.5, 4.5), facecolor="white", layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.frame_toolbar)
        toolbar.update()
        
        self.canvas.get_tk_widget().pack(expand=True)

    # --- Initialization Logic ---

    def _ui_load_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Data Folder")
        if folder_selected:
            self._set_folder(folder_selected)

    def _ui_load_phases(self):
        json_selected = filedialog.askopenfilename(title="Select Phases JSON File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if json_selected:
            try:
                with open(json_selected, 'r') as f:
                    self.phase_dict = json.load(f)
                self.cb_phase_name['values'] = list(self.phase_dict.keys())
                self.lbl_status.config(text="Phases dictionary loaded successfully.")
            except Exception as e:
                messagebox.showwarning("JSON Error", f"Could not load phases: {e}")

    def _set_folder(self, folder_path):
        """Hard reset the working directory and load all dat files."""
        self.folder = Path(folder_path)
        self.files = sorted(self.folder.rglob("*.dat"))
        
        if not self.files:
            messagebox.showerror("Error", "No .dat files found in the selected folder.")
            return False
            
        self.cache = {}
        # Notice we do NOT reset analysis_state here, so Load Project can inject it safely
        
        self.cb_samples['values'] = [f"{p.parent.name} / {p.name}" for p in self.files]
        self._load_file(0)
        self.lbl_status.config(text=f"Loaded Folder: {self.folder.name}")
        return True

    def _apply_live_settings(self, event=None):
        if not self.files: return
        try:
            w = float(self.var_ext_w.get())
            h = float(self.var_ext_h.get())
            if w < 3: w = 3
            if h < 3: h = 3
            self.fig.set_size_inches(w, h)
            self.canvas.get_tk_widget().config(width=w * self.fig.dpi, height=h * self.fig.dpi)
            self._update_plot()
        except ValueError: pass 

    def _get_current_key(self):
        if not self.files: return None
        return self.files[self.idx].name

    def _load_file(self, index):
        if not self.files: return
        self.idx = index
        self.programmatic_update = True
        self.cb_samples.current(self.idx)
        self.programmatic_update = False
        self._refresh_view()

    def _refresh_view(self):
        if self.programmatic_update or not self.files: return
        path = self.files[self.idx]
        
        if path not in self.cache:
            try:
                self.cache[path] = extract_data(path)
            except Exception as e:
                self.lbl_status.config(text=f"Error reading file: {e}")
                return

        df = self.cache[path]
        T = df["Temperature (K)"].values
        
        turn_point = np.argmax(T)
        if self.var_curve.get() == 'Heating':
            temp_h = df.iloc[15:turn_point+1].copy()
            temp_h = temp_h[temp_h["Temperature (K)"] == temp_h["Temperature (K)"].cummax()]
            self.df_view = temp_h.sort_values("Temperature (K)")
        else:
            temp_c = df.iloc[turn_point+40:-40].copy()
            temp_c = temp_c[temp_c["Temperature (K)"] == temp_c["Temperature (K)"].cummin()]
            self.df_view = temp_c.sort_values("Temperature (K)")

        if not self.df_view.empty:
            T_view = self.df_view["Temperature (K)"].values
            t_min, t_max = int(T_view.min()), int(T_view.max())
            
            for s in [self.slider_view, self.slider_r1, self.slider_r2]:
                s.update_limits(t_min, t_max)
            
            self.slider_view.set_range(t_min, t_max)
            mid = (t_min + t_max) // 2
            w = (t_max - t_min) // 10
            self.slider_r1.set_range(mid - w, mid)
            self.slider_r2.set_range(mid + w, mid + 2*w)

        key = self._get_current_key()
        if key not in self.analysis_state:
            self.analysis_state[key] = []
            
        self._populate_list()
        self._update_plot()

    def _populate_list(self):
        self.lb_phases.delete(0, tk.END)
        key = self._get_current_key()
        if not key: return
        phases = self.analysis_state.get(key, [])
        
        if not phases:
            self.var_phase_name.set("")
            self.cb_phase_name.set("")
            return

        for p in phases:
            tc_val = p.get('tc', np.nan)
            tc_str = f"{tc_val:.1f}" if not safe_isnan(tc_val) else "?"
            self.lb_phases.insert(tk.END, f"{p['name']} (Tc={tc_str}) [{p['curve']}]")
        
        self.lb_phases.selection_set(0)
        self._on_phase_select(None)

    def _add_phase(self):
        if not self.files: return
        key = self._get_current_key()
        r1 = self.slider_r1.get_range()
        r2 = self.slider_r2.get_range()
        
        new_phase = {
            "name": "New Phase",
            "r1": list(r1),
            "r2": list(r2),
            "tc": None,
            "tc_deriv": None,
            "curve": self.var_curve.get()
        }
        self.analysis_state[key].append(new_phase)
        self._populate_list()
        self.lb_phases.selection_clear(0, tk.END)
        self.lb_phases.selection_set(tk.END)
        self._on_phase_select(None)

    def _del_phase(self):
        if not self.files: return
        sel = self.lb_phases.curselection()
        if not sel: return
        idx = sel[0]
        key = self._get_current_key()
        del self.analysis_state[key][idx]
        self._populate_list()
        self._update_plot()

    def _auto_detect(self):
        if not self.files: return
        if not self.phase_dict:
            self.lbl_status.config(text="No phases loaded. Please load a phases JSON.")
            return
        
        key = self._get_current_key()
        T = self.df_view["Temperature (K)"].values
        M = self.df_view["Moment (emu)"].values
        dM = get_derivative(T, M, self.var_smooth.get())
        d2M = get_derivative(T, dM, self.var_smooth.get())
        
        count = 0
        added_indices = []

        for name, val in self.phase_dict.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                continue
            if not all(isinstance(x, (int, float)) for x in val):
                continue
            
            t_min, t_max = val
            mask = (T >= t_min) & (T <= t_max)
            if np.sum(mask) < 5: continue
            
            subset_dM = dM[mask]
            subset_T = T[mask]
            
            local_min_idx = np.argmin(subset_dM)
            inflection_T = float(subset_T[local_min_idx])
            min_dM = subset_dM[local_min_idx]
            
            exists = any(p['name'] == name and p['curve'] == self.var_curve.get() for p in self.analysis_state[key])
            if not exists:
                r1_start = float(inflection_T)
                r1_end = float(r1_start + 5.0)

                valid_tail_T = T[(T >= r1_end + 1.0) & (T <= min(t_max + 15.0, T.max()))]
                valid_tail_dM = dM[(T >= r1_end + 1.0) & (T <= min(t_max + 15.0, T.max()))]
                
                if len(valid_tail_T) > 5:
                    w = min(11, len(valid_tail_dM))
                    smoothed_tail_dM = np.convolve(valid_tail_dM, np.ones(w)/w, mode='valid')
                    closest_zero_idx = np.argmin(np.abs(smoothed_tail_dM))
                    flat_T = float(valid_tail_T[closest_zero_idx + w//2])
                    r2_start = flat_T - 2.5
                else:
                    r2_start = r1_end + 5.0
                
                r2_start = max(r2_start, r1_end + 1.0)
                r2_end = float(r2_start + 5.0)

                new_phase = {
                    "name": name,
                    "r1": [r1_start, r1_end],
                    "r2": [r2_start, r2_end],
                    "tc": float(inflection_T),
                    "tc_deriv": float(inflection_T),
                    "curve": self.var_curve.get()
                }
                self.analysis_state[key].append(new_phase)
                count += 1
                added_indices.append(len(self.analysis_state[key]) - 1)
        
        self.lbl_status.config(text=f"Auto-added {count} phases")
        self._populate_list()
        
        if added_indices:
            self.lb_phases.selection_clear(0, tk.END)
            self.lb_phases.selection_set(added_indices[0])
            self._on_phase_select(None)
        else:
            self._update_plot()

    def _on_phase_select(self, event):
        if not self.files: return
        sel = self.lb_phases.curselection()
        if not sel: return
        idx = sel[0]
        key = self._get_current_key()
        if idx >= len(self.analysis_state[key]): return
        
        p = self.analysis_state[key][idx]
        
        self.programmatic_update = True
        self.var_phase_name.set(p['name'])
        try:
            self.slider_r1.set_range(p['r1'][0], p['r1'][1])
            self.slider_r2.set_range(p['r2'][0], p['r2'][1])
        except: pass
        self.programmatic_update = False
        self._update_plot()

    def _on_name_change(self, *args):
        if self.programmatic_update or not self.files: return
        sel = self.lb_phases.curselection()
        if not sel: return
        idx = sel[0]
        key = self._get_current_key()
        self.analysis_state[key][idx]['name'] = self.var_phase_name.get()

    def _on_fit_change(self):
        if self.programmatic_update or not self.files: return
        sel = self.lb_phases.curselection()
        if not sel: return
        idx = sel[0]
        key = self._get_current_key()
        
        self.analysis_state[key][idx]['r1'] = list(self.slider_r1.get_range())
        self.analysis_state[key][idx]['r2'] = list(self.slider_r2.get_range())
        self._update_plot()

    # --- Navigation Callbacks ---
    def _on_sample_select(self, event):
        if not self.files: return
        self._load_file(self.cb_samples.current())

    def _prev_sample(self):
        if not self.files: return
        new_idx = (self.idx - 1) % len(self.files)
        self._load_file(new_idx)

    def _next_sample(self):
        if not self.files: return
        new_idx = (self.idx + 1) % len(self.files)
        self._load_file(new_idx)

    # --- Plotting ---
    
    def _update_plot(self):
        if self.df_view.empty or not self.files: return
        self.fig.clf()
        
        try: fs = float(self.var_ext_font.get())
        except ValueError: fs = 12.0
        if fs < 6: fs = 6
        
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1.5])
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
        
        key = self._get_current_key()
        t_min, t_max = self.slider_view.get_range()
        mask = (self.df_view["Temperature (K)"] >= t_min) & (self.df_view["Temperature (K)"] <= t_max)
        df_vis = self.df_view[mask]
        
        T = df_vis["Temperature (K)"].values
        M = df_vis["Moment (emu)"].values
        
        smooth_w = ensure_odd_smooth(int(self.var_smooth.get()), len(T))
        try:
            M_smooth = savgol_filter(M, smooth_w, 3)
        except Exception:
            M_smooth = M
            
        dM = get_derivative(T, M, self.var_smooth.get())
        d2M = get_derivative(T, dM, self.var_smooth.get())

        self.ax1.plot(T, M, 'k.-', alpha=0.3, markersize=2, lw=0.5)
        self.ax1.set_ylabel("Moment (emu)", fontsize=fs)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(f"{key} | {self.var_curve.get()}", fontsize=fs)
        tidy_axis(self.ax1, fs)
        
        self.ax2.plot(T, dM, '-', color='tab:blue')
        self.ax2.set_ylabel("dM/dT", fontsize=fs)
        self.ax2.grid(True, alpha=0.3)
        tidy_axis(self.ax2, fs, sci=True)
        
        self.ax3.plot(T, d2M, '-', color='tab:green')
        self.ax3.set_xlabel("Temperature (K)", fontsize=fs)
        self.ax3.set_ylabel("d²M/dT²", fontsize=fs)
        self.ax3.grid(True, alpha=0.3)
        tidy_axis(self.ax3, fs, sci=True)
        
        for ax, data in zip([self.ax2, self.ax3], [dM, d2M]):
            if len(data) > 20:
                trim_n = max(5, int(len(data) * 0.05))
                inner_data = data[trim_n:-trim_n]
                inner_data = inner_data[np.isfinite(inner_data)]
                
                if len(inner_data) > 0:
                    y_low = np.nanpercentile(inner_data, 0.2)
                    y_high = np.nanpercentile(inner_data, 99.8)
                    if y_high <= y_low: y_high = y_low + 1.0
                    margin = (y_high - y_low) * 0.1
                    ax.set_ylim(y_low - margin, y_high + margin)

        phases = self.analysis_state.get(key, [])
        sel = self.lb_phases.curselection()
        active_idx = sel[0] if sel else -1
        
        for i, p in enumerate(phases):
            if p.get('curve') != self.var_curve.get(): continue
            is_active = (i == active_idx)
            
            tc_fit, p1, p2, m_tc = fit_two_ranges_logic(T, M_smooth, p['r1'], p['r2'])
            
            mask_r1 = (T >= p['r1'][0]) & (T <= p['r1'][1])
            if np.sum(mask_r1) > 0:
                local_min_idx = np.argmin(dM[mask_r1])
                tc_deriv = T[mask_r1][local_min_idx]
            else:
                tc_deriv = np.nan

            p['tc'] = tc_fit
            p['tc_deriv'] = tc_deriv 

            base_color = get_phase_color(p['name'], 'tab:red')
            color = base_color if is_active else 'gray'
            width = 2.0 if is_active else 1.0
            
            if is_active:
                self.ax1.axvspan(p['r1'][0], p['r1'][1], color='orange', alpha=0.2)
                self.ax1.axvspan(p['r2'][0], p['r2'][1], color='green', alpha=0.2)
            
            if p1 is not None:
                x1 = np.linspace(p['r1'][0], p['r1'][1], 10)
                self.ax1.plot(x1, line(x1, *p1), color=color, lw=width, ls='--')
            if p2 is not None:
                x2 = np.linspace(p['r2'][0], p['r2'][1], 10)
                self.ax1.plot(x2, line(x2, *p2), color=color, lw=width, ls='--')
            
            # No legend: phase name + Tc are annotated directly at the transition.
            label_fit = format_phase_label(p['name'])

            if not safe_isnan(tc_fit):
                self.ax1.plot(tc_fit, m_tc, 'X', color=color, ms=7)
                annotate_tc(self.ax1, T, M, tc_fit, m_tc,
                            f"{label_fit}\n{tc_fit:.0f} K", color, fs)
                if is_active:
                    self.ax2.axvline(tc_fit, color=color, ls=':')
                    self.ax3.axvline(tc_fit, color=color, ls=':')
            
            if not safe_isnan(tc_deriv):
                self.ax2.plot(tc_deriv, dM[T == tc_deriv], 'o', color=color, markersize=4)

        add_label_headroom(self.ax1)
        self.fig.align_ylabels([self.ax1, self.ax2, self.ax3])
        self.canvas.draw_idle()


    # --- Project Save/Load ---

    def _save_session(self):
        if not self.files:
            messagebox.showwarning("Warning", "No data loaded. Start a project first.")
            return

        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Project File", "*.json")])
        if not f: return
        
        serializable = {}
        for k, v in self.analysis_state.items():
            serializable[k] = []
            for p in v:
                pc = p.copy()
                pc['tc'] = None if safe_isnan(pc.get('tc')) else float(pc['tc'])
                pc['tc_deriv'] = None if safe_isnan(pc.get('tc_deriv')) else float(pc['tc_deriv'])
                serializable[k].append(pc)
        
        # Self-contained format
        session_data = {
            "folder_path": str(self.folder.resolve()) if self.folder else "",
            "phase_dict": self.phase_dict, 
            "analysis_state": serializable
        }

        with open(f, 'w') as out:
            json.dump(session_data, out, indent=4)
        self.lbl_status.config(text=f"Project successfully saved to {Path(f).name}")

    def _load_session(self):
        f = filedialog.askopenfilename(filetypes=[("Project File", "*.json")])
        if not f: return
        
        try:
            with open(f, 'r') as inp:
                loaded = json.load(inp)
            
            if "analysis_state" in loaded:
                state = loaded["analysis_state"]
                stored_folder = loaded.get("folder_path", "")
                
                if "phase_dict" in loaded:
                    self.phase_dict = loaded["phase_dict"]
                    self.cb_phase_name['values'] = list(self.phase_dict.keys())
            else:
                state = loaded 
                stored_folder = ""

            if stored_folder and Path(stored_folder).exists():
                success = self._set_folder(stored_folder)
            elif self.folder:
                success = True 
            else:
                messagebox.showinfo("Folder Needed", "Original data folder not found. Please point me to the folder containing your .dat files.")
                fallback = filedialog.askdirectory(title="Select Folder for this Project")
                if fallback:
                    success = self._set_folder(fallback)
                else:
                    return 

            if not success: return

            for k, v in state.items():
                for p in v:
                    if safe_isnan(p.get('tc')): p['tc'] = np.nan
                    if safe_isnan(p.get('tc_deriv')): p['tc_deriv'] = np.nan
            
            self.analysis_state = state
            
            self._refresh_view()
            self.lbl_status.config(text="Project loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project: {str(e)}")

    def _export_project(self):
        if not self.files: return
        target_dir = filedialog.askdirectory(title="Select Export Folder")
        if not target_dir: return
        
        target_path = Path(target_dir) / f"Tc_Export_{datetime.date.today()}"
        target_path.mkdir(exist_ok=True)
        plots_dir = target_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        rows = []
        single_panel = bool(self.var_single_panel.get())
        
        fmt_map = {
            "SVG": ["svg"],
            "PDF": ["pdf"],
            "SVG + PNG": ["svg", "png"],
            "PNG": ["png"],
        }
        save_formats = fmt_map.get(self.var_ext_fmt.get(), ["svg"])
        primary_ext = save_formats[0]   # what the results.csv points at
        
        for fname, phases in self.analysis_state.items():
            if not phases: continue
            
            found_files = [f for f in self.files if f.name == fname]
            if not found_files: continue
            file_path = found_files[0]
            
            try:
                folder_source = str(file_path.parent.relative_to(self.folder))
            except ValueError:
                folder_source = file_path.parent.name
            
            try:
                df = extract_data(file_path)
                T = df["Temperature (K)"].values
                M = df["Moment (emu)"].values
                turn = np.argmax(T)
                
                phases_by_curve = {}
                for p in phases:
                    c = p.get('curve', 'Heating')
                    if c not in phases_by_curve: phases_by_curve[c] = []
                    phases_by_curve[c].append(p)
                
                for c_type, p_list in phases_by_curve.items():
                    if c_type == 'Heating':
                        temp_h = df.iloc[15:turn+1].copy()
                        temp_h = temp_h[temp_h["Temperature (K)"] == temp_h["Temperature (K)"].cummax()]
                        df_slice = temp_h.sort_values("Temperature (K)")
                    else:
                        temp_c = df.iloc[turn+40:-40].copy()
                        temp_c = temp_c[temp_c["Temperature (K)"] == temp_c["Temperature (K)"].cummin()]
                        df_slice = temp_c.sort_values("Temperature (K)")
                        
                    if df_slice.empty: continue
                    
                    Ts = df_slice["Temperature (K)"].values
                    Ms = df_slice["Moment (emu)"].values
                    
                    w = float(self.var_ext_w.get())
                    h = float(self.var_ext_h.get())
                    fs = float(self.var_ext_font.get())
                    if fs < 6: fs = 6
                    
                    exp_fig = Figure(figsize=(w, h), facecolor='white', layout='constrained')
                    
                    if single_panel:
                        eax1 = exp_fig.add_subplot(1, 1, 1)
                        eax2 = eax3 = None
                    else:
                        gs = exp_fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1.5])
                        eax1 = exp_fig.add_subplot(gs[0])
                        eax2 = exp_fig.add_subplot(gs[1], sharex=eax1)
                        eax3 = exp_fig.add_subplot(gs[2], sharex=eax1)
                    
                    eax1.plot(Ts, Ms, 'k.-', alpha=0.3, markersize=2, lw=0.5)
                    
                    dM = get_derivative(Ts, Ms, self.var_smooth.get())
                    d2M = get_derivative(Ts, dM, self.var_smooth.get())
                    
                    if not single_panel:
                        eax2.plot(Ts, dM, '-', color='tab:blue')
                        eax3.plot(Ts, d2M, '-', color='tab:green')
                        
                        for ax, data in zip([eax2, eax3], [dM, d2M]):
                            if len(data) > 20:
                                trim_n = max(5, int(len(data) * 0.05))
                                inner_data = data[trim_n:-trim_n]
                                inner_data = inner_data[np.isfinite(inner_data)]
                                if len(inner_data) > 0:
                                    y_low = np.nanpercentile(inner_data, 0.2)
                                    y_high = np.nanpercentile(inner_data, 99.8)
                                    if y_high <= y_low: y_high = y_low + 1.0
                                    margin = (y_high - y_low) * 0.1
                                    ax.set_ylim(y_low - margin, y_high + margin)
                    
                    comp_name = fname.split('_')[0].replace('.dat', '')
                    elements = parse_formula(comp_name)
                    
                    for i, p in enumerate(p_list):
                        tc_fit, p1, p2, m_tc = fit_two_ranges_logic(Ts, Ms, p['r1'], p['r2'])
                        tc_deriv = p.get('tc_deriv', np.nan)
                        color = get_phase_color(p['name'], f"C{i}")
                        
                        if p1 is not None:
                            x1 = np.linspace(p['r1'][0], p['r1'][1], 10)
                            eax1.plot(x1, line(x1, *p1), color=color, ls='--')
                        if p2 is not None:
                            x2 = np.linspace(p['r2'][0], p['r2'][1], 10)
                            eax1.plot(x2, line(x2, *p2), color=color, ls='--')
                        
                        # No legend: phase name + Tc are annotated at the transition.
                        label_fit = format_phase_label(p['name'])
                        
                        if not safe_isnan(tc_fit):
                            eax1.plot(tc_fit, m_tc, 'X', color=color, ms=7)
                            annotate_tc(eax1, Ts, Ms, tc_fit, m_tc,
                                        f"{label_fit}\n{tc_fit:.0f} K", color, fs)
                            if not single_panel:
                                eax2.axvline(tc_fit, color=color, ls=':')
                                eax3.axvline(tc_fit, color=color, ls=':')
                        
                        if not single_panel and not safe_isnan(tc_deriv):
                            if len(dM) > 0 and len(Ts) > 0 and np.sum(Ts == tc_deriv) > 0:
                                eax2.plot(tc_deriv, dM[Ts == tc_deriv], 'o', color=color, markersize=4)
                        
                        plot_filename = f"{fname.replace('.dat', '')}_{c_type}.{primary_ext}"
                        row = {
                            "File": fname,
                            "Folder_Source": folder_source,
                            "Nominal_Composition": comp_name,
                            "Phase": p['name'],
                            "Tc_Fit (K)": tc_fit,
                            "Tc_Deriv (K)": tc_deriv,
                            "Curve": c_type,
                            "Image_Path": f"plots/{plot_filename}"
                        }
                        for k, v in elements.items(): row[k] = v
                        rows.append(row)
                    
                    eax1.set_ylabel("Moment (emu)", fontsize=fs)
                    add_label_headroom(eax1)
                    eax1.grid(True, alpha=0.3)
                    tidy_axis(eax1, fs)
                    
                    if single_panel:
                        eax1.set_xlabel("Temperature (K)", fontsize=fs)
                    else:
                        eax2.set_ylabel("dM/dT", fontsize=fs)
                        eax2.grid(True, alpha=0.3)
                        tidy_axis(eax2, fs, sci=True)
                        
                        eax3.set_xlabel("Temperature (K)", fontsize=fs)
                        eax3.set_ylabel("d²M/dT²", fontsize=fs)
                        eax3.grid(True, alpha=0.3)
                        tidy_axis(eax3, fs, sci=True)
                        exp_fig.align_ylabels([eax1, eax2, eax3])
                    
                    stem = f"{fname.replace('.dat', '')}_{c_type}"
                    for ext in save_formats:
                        # No bbox_inches='tight': the file keeps its exact
                        # requested size, so the font size is true when placed
                        # in Word at 100% scale.
                        exp_fig.savefig(plots_dir / f"{stem}.{ext}",
                                        dpi=600 if ext == 'png' else None,
                                        format=ext)
                    exp_fig.clf() 

            except Exception as e:
                print(f"Skipping {fname}: {e}")
                continue

        if rows:
            csv_path = target_path / "results.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            messagebox.showinfo("Export Complete", f"Project exported to:\n{target_path}")
            self.lbl_status.config(text=f"Exported to {target_path.name}")
        else:
            messagebox.showinfo("Export Empty", "No analyzed phases found to export.")

    def _merge_sem_data(self):
            # 1. Ask the user for the MT Results CSV they just exported
            mt_file = filedialog.askopenfilename(
                title="Step 1: Select MT Export (results.csv)", 
                filetypes=[("CSV Files", "*.csv")]
            )
            if not mt_file: return
            
            # 2. Ask the user for the SEM Excel File
            sem_file = filedialog.askopenfilename(
                title="Step 2: Select SEM Export (.xlsx)", 
                filetypes=[("Excel Files", "*.xlsx")]
            )
            if not sem_file: return
            
            try:
                # Load the MT data
                df_mt = pd.read_csv(mt_file)
                
                # Load the SEM Excel file to get the correct sheets
                xls = pd.ExcelFile(sem_file)
                if 'Summary' not in xls.sheet_names or 'Phase_Avgs' not in xls.sheet_names:
                    messagebox.showerror(
                        "Merge Error", 
                        "The selected SEM Excel file must contain both 'Summary' and 'Phase_Avgs' sheets."
                    )
                    return
                    
                df_sem_summary = pd.read_excel(xls, sheet_name='Summary')
                df_sem_phase_avgs = pd.read_excel(xls, sheet_name='Phase_Avgs')
                
                # Filter out AVERAGE rows from Summary to avoid pollution in the raw merge
                df_sem_summary_clean = df_sem_summary[df_sem_summary['File'] != 'AVERAGE']
                
                # Verify primary keys exist
                if 'Folder_Source' not in df_mt.columns or 'Sample_Folder' not in df_sem_summary.columns:
                    messagebox.showerror(
                        "Merge Error", 
                        "Missing matching columns. MT needs 'Folder_Source' and SEM needs 'Sample_Folder'."
                    )
                    return

                # Ask user for the output file immediately to abort early if they cancel
                out_file = filedialog.asksaveasfilename(
                    title="Save ML Ready Database", 
                    defaultextension=".xlsx", 
                    filetypes=[("Excel", "*.xlsx")]
                )
                if not out_file: return

                # ==========================================
                # GENERATE DIAGNOSTICS (The "Human Error" Detector)
                # ==========================================
                mt_samples = set(df_mt['Folder_Source'].dropna().astype(str).unique())
                sem_samples = set(df_sem_summary_clean['Sample_Folder'].dropna().astype(str).unique())
                
                mt_phases = set(df_mt['Phase'].dropna().astype(str).unique())
                sem_phases = set(df_sem_phase_avgs['Phase'].dropna().astype(str).unique())
                
                audit_rows = []
                
                for s in (mt_samples - sem_samples):
                    audit_rows.append({'Issue Type': 'Orphan Sample (MT)', 'Value': s, 'Description': 'Found in MT export, but no exact matching folder name in SEM Excel.'})
                for s in (sem_samples - mt_samples):
                    audit_rows.append({'Issue Type': 'Orphan Sample (SEM)', 'Value': s, 'Description': 'Found in SEM Excel, but no exact matching folder name in MT export.'})
                    
                for p in (mt_phases - sem_phases):
                    audit_rows.append({'Issue Type': 'Unmatched Phase Naming (MT)', 'Value': p, 'Description': 'Phase name exists in MT data, but was never detected or named identically in the SEM data.'})
                for p in (sem_phases - mt_phases):
                    audit_rows.append({'Issue Type': 'Unmatched Phase Naming (SEM)', 'Value': p, 'Description': 'Phase name exists in SEM data, but is not defined in the MT data.'})

                df_audit = pd.DataFrame(audit_rows)
                if df_audit.empty:
                    df_audit = pd.DataFrame([{'Issue Type': 'All Clear', 'Value': 'N/A', 'Description': 'No typos or naming mismatches detected between MT and SEM.'}])

                with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
                    
                    # Write the Diagnostics Sheet First
                    df_audit.to_excel(writer, sheet_name='0_Merge_Diagnostics', index=False)

                    # ==========================================
                    # SHEET 1: Master Raw Merge
                    # ==========================================
                    df_master = pd.merge(
                        df_mt, 
                        df_sem_summary_clean, 
                        left_on='Folder_Source', 
                        right_on='Sample_Folder', 
                        how='left',
                        suffixes=('_MT', '_SEM')
                    )
                    df_master.to_excel(writer, sheet_name='1_Master_Raw', index=False)

                    # ==========================================
                    # SHEET 2: Phase Composition and Tc
                    # ==========================================
                    # Average the MT data per sample and phase
                    mt_phase = df_mt.groupby(['Folder_Source', 'Phase']).agg({
                        'Tc_Fit (K)': 'mean',
                        'Tc_Deriv (K)': 'mean'
                    }).reset_index()

                    # Average the SEM phase data (from the Phase_Avgs sheet)
                    sem_num_cols = df_sem_phase_avgs.select_dtypes(include=[np.number]).columns.tolist()
                    elements = [c for c in sem_num_cols if c not in ['Count', 'Unnamed: 0']]
                    
                    sem_phase = df_sem_phase_avgs.groupby(['Sample_Folder', 'Phase'])[elements].mean().reset_index()

                    # Merge them on Sample and Phase (Outer join to expose missing links)
                    df_phase_ml = pd.merge(
                        mt_phase, 
                        sem_phase, 
                        left_on=['Folder_Source', 'Phase'], 
                        right_on=['Sample_Folder', 'Phase'], 
                        how='outer'
                    )
                    
                    # Consolidate Sample IDs
                    df_phase_ml['Sample_ID'] = df_phase_ml['Folder_Source'].fillna(df_phase_ml['Sample_Folder'])
                    df_phase_ml['Phase_ID'] = df_phase_ml['Phase_x'].fillna(df_phase_ml['Phase_y']) if 'Phase_x' in df_phase_ml.columns else df_phase_ml['Phase']
                    
                    # Clean up duplicate phase columns from outer merge
                    drop_cols = ['Folder_Source', 'Sample_Folder']
                    if 'Phase_x' in df_phase_ml.columns:
                        drop_cols.extend(['Phase_x', 'Phase_y'])
                    else:
                        drop_cols.append('Phase')
                        
                    # Reorder columns cleanly
                    cols_s2 = ['Sample_ID', 'Phase_ID', 'Tc_Fit (K)', 'Tc_Deriv (K)'] + [c for c in df_phase_ml.columns if c not in (['Sample_ID', 'Phase_ID', 'Tc_Fit (K)', 'Tc_Deriv (K)'] + drop_cols)]
                    df_phase_ml[cols_s2].rename(columns={'Phase_ID': 'Phase'}).to_excel(writer, sheet_name='2_Phase_Comp_and_Tc', index=False)

                    # ==========================================
                    # SHEET 3: Sample Cross Validation
                    # ==========================================
                    # String together unique phases found by MT
                    mt_sample_phases = df_mt.groupby('Folder_Source')['Phase'].apply(
                        lambda x: ', '.join(sorted(set(x.dropna().astype(str))))
                    ).reset_index(name='MT_Detected_Phases')

                    # String together unique phases found by SEM
                    sem_sample_phases = df_sem_phase_avgs.groupby('Sample_Folder')['Phase'].apply(
                        lambda x: ', '.join(sorted(set(x.dropna().astype(str))))
                    ).reset_index(name='SEM_Detected_Phases')

                    # Average the bulk area composition per sample from the clean summary
                    sem_bulk_avg = df_sem_summary_clean.groupby('Sample_Folder').mean(numeric_only=True).reset_index()
                    
                    # Merge Cross Validation Data
                    df_sample_val = pd.merge(mt_sample_phases, sem_sample_phases, left_on='Folder_Source', right_on='Sample_Folder', how='outer')
                    
                    # Fill missing sample names from either side
                    df_sample_val['Sample_ID'] = df_sample_val['Folder_Source'].fillna(df_sample_val['Sample_Folder'])
                    df_sample_val = df_sample_val.drop(columns=['Folder_Source', 'Sample_Folder'])
                    
                    # Attach bulk composition
                    df_sample_ml = pd.merge(df_sample_val, sem_bulk_avg, left_on='Sample_ID', right_on='Sample_Folder', how='left')
                    if 'Sample_Folder' in df_sample_ml.columns:
                        df_sample_ml = df_sample_ml.drop(columns=['Sample_Folder'])
                    
                    # Reorder columns
                    cols_s3 = ['Sample_ID', 'MT_Detected_Phases', 'SEM_Detected_Phases'] + [c for c in df_sample_ml.columns if c not in ['Sample_ID', 'MT_Detected_Phases', 'SEM_Detected_Phases']]
                    df_sample_ml[cols_s3].to_excel(writer, sheet_name='3_Sample_CrossVal', index=False)
                    
                # Alert the user if human errors were found
                if not df_audit[df_audit['Issue Type'] != 'All Clear'].empty:
                    err_count = len(df_audit)
                    messagebox.showwarning(
                        "Merge Completed with Warnings", 
                        f"Merge successful, but {err_count} naming mismatches were found between MT and SEM data.\n\nPlease check the '0_Merge_Diagnostics' sheet in the exported Excel file."
                    )
                else:
                    messagebox.showinfo("Success", f"Perfect match! ML ready workbook saved to:\n{Path(out_file).name}")
                
            except ImportError:
                messagebox.showerror("Dependency Error", "Please run 'pip install openpyxl' to save Excel files.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to merge databases: {str(e)}")


if __name__ == "__main__":
    if THEME_AVAILABLE:
        root = tb.Window(themename="superhero")
    else:
        root = tk.Tk()
        
    app = TcNavigatorApp(root)
    root.mainloop()


