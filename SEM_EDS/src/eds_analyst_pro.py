# %%
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import pandas as pd
import numpy as np
import json, threading, os, re, logging

try:
    import fitz
except ImportError:
    fitz = None

import matplotlib
matplotlib.use("Agg")          # figures are only saved to disk, never shown
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# =========================================================================
# PANEL SETTINGS - keep identical to the MT/Tc Navigator app so that the
# lettering matches when both figures sit in the same composite panel.
# =========================================================================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

PANEL_FIG_W = 6.5     # inches - MUST match "Fig W" in the MT/Tc app
PANEL_FONT = 14.0     # points  - MUST match "Font" in the MT/Tc app
PANEL_DPI = 600       # raster resolution of the map itself


def make_map_figure(data, cmap, cbar_label, y_max, w,
                    fig_w=PANEL_FIG_W, fs=PANEL_FONT, panel_label=None):
    """
    Build a map figure whose physical width and font size match the MT/Tc
    figures, so both render at the same lettering size when placed in one
    panel at 100% scale.

    panel_label: optional 'a', 'b', ... drawn bold at the top-left in the
    Nature Portfolio panel style.
    """
    aspect = float(y_max) / float(w) if w else 0.75
    fig_h = fig_w * aspect + 1.0          # room for the x-label and ticks
    fig_h = max(2.5, min(fig_h, 9.0))     # stay sane for odd aspect ratios

    fig = Figure(figsize=(fig_w, fig_h), facecolor='white', layout='constrained')
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    cbar.outline.set_linewidth(0.8)

    # Sentence case, no bold: Nature Portfolio house style.
    ax.set_xlabel('Distance (pixels)', fontsize=fs)
    ax.set_ylabel('Distance (pixels)', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    if panel_label:
        ax.text(0.0, 1.02, panel_label, transform=ax.transAxes,
                fontsize=fs * 1.15, fontweight='bold', ha='left', va='bottom')
    return fig


ATOMIC_NUMBERS = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Pb':82,'Bi':83,'Th':90,'U':92}
RE_ELEMENTS = {'Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'}
TM_ELEMENTS = {'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zr','Nb','Mo','Hf','Ta','W'}
PHASE_COLORS = [(255,80,80),(80,255,80),(80,100,255),(255,255,80),(0,255,255),(255,80,255)]

# Fallback colours, only used when ttkbootstrap is unavailable. When the theme
# is present these are replaced at runtime by the real palette (see _apply_palette).
CANVAS_BG = "#1e1e1e"
PANEL_BG = "#2b2b2b"
PANEL_FG = "#ffffff"

# -------------------------------------------------------------------------
# Theme bootstrap
# -------------------------------------------------------------------------
THEME_AVAILABLE = False
try:
    import ttkbootstrap as tb
    try:
        # canonical location in ttkbootstrap >= 1.0
        from ttkbootstrap.scrolled import ScrolledFrame
    except ImportError:
        # older layout
        from ttkbootstrap.widgets.scrolled import ScrolledFrame
    THEME_AVAILABLE = True
except ImportError:
    tb = tk

    class ScrolledFrame(tk.Frame):
        def __init__(self, p, width=300, **kw):
            super().__init__(p, **kw)
            self.c = tk.Canvas(self, borderwidth=0, background=PANEL_BG,
                               highlightthickness=0)
            self.v = tk.Frame(self.c, background=PANEL_BG)
            self.s = ttk.Scrollbar(self, orient="vertical", command=self.c.yview)
            self.c.configure(yscrollcommand=self.s.set)
            self.s.pack(side="right", fill="y")
            self.c.pack(side="left", fill="both", expand=True)
            self.c.create_window((4, 4), window=self.v, anchor="nw", tags="self.v")
            self.v.bind("<Configure>",
                        lambda e: self.c.configure(scrollregion=self.c.bbox("all")))
            self.c.bind("<Configure>",
                        lambda e: self.c.itemconfig("self.v", width=e.width))
            self.view_port = self.v


try:
    from ttkbootstrap.tooltip import ToolTip
except Exception:
    ToolTip = None


def create_button(p, t, c, s=None, **kw):
    return tb.Button(p, text=t, command=c, bootstyle=s, **kw) \
        if THEME_AVAILABLE and s else ttk.Button(p, text=t, command=c, **kw)


def create_check(p, var, cmd=None, s=None):
    if THEME_AVAILABLE and s:
        return tb.Checkbutton(p, variable=var, command=cmd, bootstyle=s)
    return ttk.Checkbutton(p, variable=var, command=cmd)


def tip(widget, text):
    """Attach a tooltip when ttkbootstrap provides one; silently skip otherwise."""
    if ToolTip is not None:
        try:
            ToolTip(widget, text=text, delay=400)
        except Exception:
            pass


# ---- small colour helpers, used to derive a consistent palette from the theme ----
def _hex_to_rgb(h):
    h = str(h).lstrip('#')
    if len(h) == 3: h = ''.join(ch * 2 for ch in h)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r, g, b):
    return '#%02x%02x%02x' % (max(0, min(255, int(r))),
                              max(0, min(255, int(g))),
                              max(0, min(255, int(b))))


def shade(hex_color, factor):
    """factor < 1 darkens toward black, > 1 lightens toward white."""
    try:
        r, g, b = _hex_to_rgb(hex_color)
    except Exception:
        return hex_color
    if factor <= 1:
        return _rgb_to_hex(r * factor, g * factor, b * factor)
    f = factor - 1
    return _rgb_to_hex(r + (255 - r) * f, g + (255 - g) * f, b + (255 - b) * f)


ML_AVAILABLE = False
try:
    from skimage import filters, color
    from skimage.filters import threshold_multiotsu
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    pass


# =========================================================================
# Themed replacement for tk.Listbox.
# tk.Listbox is a classic Tk widget and ignores the ttk theme entirely, which
# is what made the Spots panel render white/grey. A Treeview in "tree" mode
# looks identical but is fully themed. The API below mirrors the Listbox
# methods this app actually used, so calling code stays unchanged.
# =========================================================================
class ThemedListbox(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.tv = ttk.Treeview(self, show="tree", selectmode="browse")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._items = []

    def insert(self, index, value):
        self._items.append(self.tv.insert("", "end", text=str(value)))

    def delete(self, first=0, last=None):
        for i in self.tv.get_children():
            self.tv.delete(i)
        self._items = []

    def curselection(self):
        sel = self.tv.selection()
        if not sel or sel[0] not in self._items:
            return ()
        return (self._items.index(sel[0]),)

    def get(self, index):
        try:
            return self.tv.item(self._items[index], "text")
        except (IndexError, tk.TclError):
            return ""

    def size(self):
        return len(self._items)

    def select_set(self, index):
        if 0 <= index < len(self._items):
            self.tv.selection_set(self._items[index])
            self.tv.focus(self._items[index])

    def bind_select(self, callback):
        self.tv.bind("<<TreeviewSelect>>", callback)


class ZoomableImageCanvas(tk.Canvas):
    def __init__(self, p, bg_color=CANVAS_BG, **kw):
        super().__init__(p, bg=bg_color, highlightthickness=0, **kw)
        self.pil_image, self.tk_image = None, None
        self.scale, self.pan_x, self.pan_y = 1.0, 0, 0
        self._drag = {"x": 0, "y": 0}
        self.bind("<ButtonPress-2>", self.start_pan); self.bind("<B2-Motion>", self.pan)
        self.bind("<ButtonPress-3>", self.start_pan); self.bind("<B3-Motion>", self.pan)
        self.bind("<MouseWheel>", self.zoom); self.bind("<Button-4>", self.zoom); self.bind("<Button-5>", self.zoom)
        self.bind("<Configure>", self.draw)

    def set_image(self, img):
        if not img: return
        self.pil_image = img.convert("RGB")
        self.scale, self.pan_x, self.pan_y = 1.0, 0, 0
        self.draw()

    def draw(self, e=None):
        if not self.winfo_exists() or not self.pil_image: return
        w, h = int(self.pil_image.width * self.scale), int(self.pil_image.height * self.scale)
        if w < 1 or h < 1: return
        self.tk_image = ImageTk.PhotoImage(self.pil_image.resize((w, h), Image.Resampling.BILINEAR))
        self.delete("all")
        self.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image)

    def to_coords(self, x, y): return (x - self.pan_x)/self.scale, (y - self.pan_y)/self.scale

    def start_pan(self, e): self._drag["x"], self._drag["y"] = e.x, e.y

    def pan(self, e):
        self.pan_x += e.x - self._drag["x"]; self.pan_y += e.y - self._drag["y"]
        self._drag["x"], self._drag["y"] = e.x, e.y; self.draw()

    def zoom(self, e):
        if not self.pil_image: return
        f = 1.1 if (e.delta > 0 or e.num == 4) else 0.9
        self.pan_x = e.x - (e.x - self.pan_x) * f; self.pan_y = e.y - (e.y - self.pan_y) * f
        self.scale *= f; self.draw()


def load_sem_image_robust(path):
    try:
        img = Image.open(path)
        if img.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            a = np.array(img); mn, mx = a.min(), a.max()
            a = (a - mn)/(mx - mn)*255 if mx > mn else np.zeros_like(a)
            img = Image.fromarray(a.astype('uint8'))
        return img.convert("RGB")
    except Exception:
        return None


def get_z(comp):
    tz, ta = 0.0, 0.0
    for e, p in comp.items(): tz += ATOMIC_NUMBERS.get(e,0)*p; ta += p
    return tz/ta if ta>0 else 0


def est_z(n):
    if any(x in n.lower() for x in ["void","pore","back","resin"]): return 0.0
    m = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', n)
    if not m: return 0.0
    tz, ta = 0.0, 0.0
    for e, c in m:
        z = ATOMIC_NUMBERS.get(e, 0)
        if z==0: continue
        v = float(c) if c else 1.0
        tz+=z*v; ta+=v
    return tz/ta if ta>0 else 0


def suggest(comp, rules):
    for p in rules:
        m = True
        for r in p.get('rules', []):
            if r.get('type','element') == 'element':
                v = comp.get(r.get('target') or r.get('element'), 0.0)
                if not (r['min'] <= v <= r['max']): m = False; break
            elif r.get('type') == 'ratio':
                n = sum(comp.get(e,0) for e in r.get('numerator',[]))
                d = sum(comp.get(e,0) for e in r.get('denominator',[]))
                if d<0.1 or not (r['min'] <= n/d <= r['max']): m = False; break
        if m: return p['name']
    return ""


def parse_eds(path):
    try: df = pd.read_excel(path, header=None)
    except Exception as e: return None, str(e)
    meas, spots = {}, []
    ecol, acol, hrow = None, None, None
    for i, r in df.iterrows():
        s = r.astype(str).values
        if 'Element' in s and any(x in str(s) for x in ['Atomic %', 'At%']):
            hrow = i
            for j, c in enumerate(r):
                if str(c).strip()=='Element': ecol=j
                if str(c).strip() in ['Atomic %','At%','Atomic Percent']: acol=j
            break
    if ecol is None: return None, "No Header"
    for i, r in df.iterrows():
        m = re.search(r'(EDS Spot \d+|Spectrum \d+|Selected Area \d+|Spot \d+)',
                      " ".join([str(x) for x in r]), re.IGNORECASE)
        if m:
            n = m.group(1)
            if not any(x[0]==n for x in spots): spots.append((n, i))
    for k, (n, start) in enumerate(spots):
        meas[n] = {"composition": {}, "phase": "", "excluded_elements": []}
        end = spots[k+1][1] if k+1 < len(spots) else len(df)
        sub = df.iloc[(hrow+1 if k==0 else start+1):end]
        for _, r in sub.iterrows():
            if pd.isna(r.iloc[ecol]): continue
            rel, val = str(r.iloc[ecol]).strip(), r.iloc[acol]
            if str(val).lower()=='nan' or "total" in rel.lower(): continue
            try: meas[n]["composition"][re.match(r"^([A-Z][a-z]?)", rel).group(1)] = float(val)
            except Exception: continue
    return meas, None


if ML_AVAILABLE:
    MODEL_FORMAT = "eds-analyst-model"

    class Segmenter:
        def __init__(self):
            self.m = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=25))
            self.iforest = make_pipeline(StandardScaler(), IsolationForest(n_estimators=50, contamination=0.05, n_jobs=-1))
            self.ok = False          # random forest is fitted
            self.iso_ok = False      # isolation forest is fitted

        def feats(self, img):
            g = color.rgb2gray(img) if len(img.shape)==3 else img.astype(float)/255.0
            f = [g.reshape(-1, 1)]
            for s in [1, 2, 4]: f.append(filters.gaussian(g, sigma=s).reshape(-1, 1))
            f.append(filters.sobel(g).reshape(-1, 1))
            return np.hstack(f)

        def train(self, img, lbl, roi=None, n_est=50):
            self.m = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_est, n_jobs=-1, max_depth=25))
            self.iforest = make_pipeline(StandardScaler(), IsolationForest(n_estimators=n_est, contamination=0.02, n_jobs=-1))
            self.ok = self.iso_ok = False

            X, Y = self.feats(img), lbl.reshape(-1)
            if roi is not None: Y[~roi.reshape(-1)] = 0
            m = Y > 0
            if len(Y[m]) < 50: raise ValueError("Paint more!")

            self.m.fit(X[m], Y[m])
            self.iforest.fit(X[m])   # anomaly detector trained ONLY on brushed pixels
            self.ok = True
            self.iso_ok = True

        # ---- persistence -------------------------------------------------
        def bundle(self, **extra):
            """Everything needed to reproduce this model on another image."""
            b = {"format": MODEL_FORMAT, "version": 2,
                 "rf": self.m,
                 "iforest": self.iforest if self.iso_ok else None}
            b.update(extra)
            return b

        def load_bundle(self, obj):
            """Accepts both the v2 bundle and a bare v1 pipeline.
            Returns True when an anomaly detector came with the file."""
            if isinstance(obj, dict) and obj.get("format") == MODEL_FORMAT:
                self.m = obj["rf"]
                self.ok = True
                iso = obj.get("iforest")
                if iso is not None:
                    self.iforest = iso
                    self.iso_ok = True
                else:
                    self.iso_ok = False
            else:
                # Old file: random forest only, no anomaly detector inside.
                self.m = obj
                self.ok = True
                self.iso_ok = False
            return self.iso_ok

        # ---- inference ---------------------------------------------------
        def _anomaly_from_X(self, X):
            # Raw scores: lower = more anomalous
            risk = -self.iforest.decision_function(X)

            r_min, r_max = risk.min(), risk.max()
            if r_max > r_min:
                risk = (risk - r_min) / (r_max - r_min)
            else:
                risk = np.zeros_like(risk)

            # Squaring boosts contrast: normal pixels go transparent, anomalies pop.
            return risk ** 2

        def predict_all(self, img, roi=None):
            """Labels, reliability and anomaly risk from ONE feature pass.

            The feature stack (three gaussians plus a sobel over the whole
            image) is the expensive part, so computing it once instead of
            three times makes this roughly 3x faster than calling the
            individual predictors in sequence.
            Anomaly is None when no fitted isolation forest is available.
            """
            if not self.ok: return None, None, None
            X = self.feats(img)
            shape = img.shape[:2]

            lbl = self.m.predict(X).reshape(shape)
            conf = np.max(self.m.predict_proba(X), axis=1).reshape(shape)
            anom = self._anomaly_from_X(X).reshape(shape) if self.iso_ok else None

            if roi is not None:
                lbl[~roi] = 0
                conf[~roi] = 0
                if anom is not None: anom[~roi] = 0
            return lbl, conf, anom

        def predict(self, img, roi=None):
            if not self.ok: return None
            r = self.m.predict(self.feats(img)).reshape(img.shape[:2])
            if roi is not None: r[~roi] = 0
            return r

        def predict_confidence(self, img, roi=None):
            if not self.ok: return None
            p = self.m.predict_proba(self.feats(img))
            conf = np.max(p, axis=1).reshape(img.shape[:2])
            if roi is not None: conf[~roi] = 0
            return conf

        def predict_anomaly(self, img, roi=None):
            # iso_ok guards against an unfitted forest, which is what a model
            # loaded from an old .joblib leaves behind.
            if not self.ok or not self.iso_ok: return None
            conf = self._anomaly_from_X(self.feats(img)).reshape(img.shape[:2])
            if roi is not None: conf[~roi] = 0
            return conf


class EDSBatchAnalyzerApp(tb.Window if THEME_AVAILABLE else tk.Tk):
    def __init__(self):
        if THEME_AVAILABLE:
            try:
                if hasattr(tb.Style, '_instance'):
                    tb.Style._instance = None
            except Exception:
                pass

        super().__init__(themename="superhero") if THEME_AVAILABLE else super().__init__()

        if THEME_AVAILABLE:
            try:
                self.style.theme_use('superhero')
            except Exception:
                pass

        self.title("Microstructure EDS Analyst Pro")

        # -----------------------------------------------------------------
        # Adaptive window: size to the screen instead of a hard 1800x950.
        # -----------------------------------------------------------------
        self._setup_geometry()
        self._apply_palette()
        self._style_widgets()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.proj, self.seg_data, self.cur_file, self.cur_spot = {}, {}, None, None
        self.project_masks = {}

        try: self.rules = json.load(open("phases_config.json"))
        except Exception: self.rules = [{"name": "Matrix"}, {"name": "Precipitate"}]

        base_phases = [p['name'] for p in self.rules]
        if "Void" not in base_phases: base_phases.append("Void")
        self.phases = list(dict.fromkeys(base_phases))

        self.pmap = {n: i+1 for i, n in enumerate(self.phases)}
        self.imap = {v: k for k, v in self.pmap.items()}

        base_cols = [
            (255, 80, 80), (80, 255, 80), (80, 100, 255), (255, 255, 80),
            (0, 255, 255), (255, 80, 255), (255, 165, 0), (128, 0, 128),
            (0, 128, 128), (128, 128, 0), (0, 0, 128), (139, 69, 19)
        ]
        self.colors = [c + (120,) for c in base_cols]

        self.img_path, self.pil, self.arr, self.lbl, self.res = None, None, None, None, None
        self.roi, self.rstart, self.undo_s = None, None, []
        self.mode, self.drawing, self.last_m = "brush", False, (0, 0)

        self.cut = tk.DoubleVar(value=20.6)
        self.brush_sz = tk.IntVar(value=5)
        self.brt = tk.DoubleVar(value=1.0)
        self.cnt = tk.DoubleVar(value=1.0)
        self.ovl = tk.BooleanVar(value=True)

        self.p_seg = tk.StringVar(value=self.phases[0])
        self.p_exc = tk.StringVar(value="(None)")
        self.st_var = tk.StringVar(value="Ready")
        self.p_eds = tk.StringVar()

        # Single source of truth for the overlay mode (it used to be created
        # twice - once here as "Segmentation" and again in ui() as "ML Mask").
        self.ovl_mode = tk.StringVar(value="ML Mask")
        self.conf_map = None
        self.anomaly_map = None

        # Export geometry - keep in sync with the MT/Tc Navigator app
        self.fig_w_var = tk.DoubleVar(value=PANEL_FIG_W)
        self.fig_font_var = tk.DoubleVar(value=PANEL_FONT)
        self.n_est = tk.IntVar(value=50)

        self.p_vis = {p: tk.BooleanVar(value=True) for p in self.phases}
        self.p_auto = {p: tk.BooleanVar(value=False) for p in self.phases}
        if ML_AVAILABLE: self.ml = Segmenter()

        i = Image.new('RGBA', (16, 16), (0, 0, 0, 0)); d = ImageDraw.Draw(i)
        d.rectangle((3, 3, 12, 12), outline='white', width=1)
        self.i_off = ImageTk.PhotoImage(i)
        d.line((5, 5, 10, 10), fill='#00ff00', width=2); d.line((5, 10, 10, 5), fill='#00ff00', width=2)
        self.i_on = ImageTk.PhotoImage(i)

        self.ui()

    # ---------------------------------------------------------------------
    # Window sizing / theming helpers
    # ---------------------------------------------------------------------
    def _setup_geometry(self):
        """Size the window to a fraction of the actual screen and centre it."""
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        w = int(sw * 0.92)
        h = int(sh * 0.88)

        # Never demand more than the screen, never go below a workable minimum.
        w = max(1100, min(w, sw - 40))
        h = max(700, min(h, sh - 80))

        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 20)

        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1100, 700)

        # Remembered for the initial sash placement.
        self._win_w = w

    def _apply_palette(self):
        """Build one palette for the whole app, derived from the active theme.

        Surfaces are three tones of the same blue rather than white:
            canvas  (darkest)  image areas
            field   (dark)     tables, entries, spinboxes, lists
            bg      (mid)      panels and frames
            head    (light)    table headings
        """
        global CANVAS_BG, PANEL_BG, PANEL_FG
        pal = dict(bg='#2b3e50', fg='#e6edf3', accent='#4c9be8',
                   ok='#5cb85c', warn='#f0ad4e', bad='#d9534f')
        if THEME_AVAILABLE:
            try:
                c = self.style.colors
                pal['bg'] = c.bg
                pal['fg'] = c.fg
                pal['accent'] = c.primary
                pal['ok'] = c.success
                pal['warn'] = c.warning
                pal['bad'] = c.danger
            except Exception:
                pass

        bg = pal['bg']
        self.C = {
            'bg':     bg,
            'field':  shade(bg, 0.70),
            'canvas': shade(bg, 0.42),
            'head':   shade(bg, 1.14),
            'border': shade(bg, 0.52),
            'stripe': shade(bg, 0.78),
            'fg':     pal['fg'],
            'muted':  shade(pal['fg'], 0.60),
            'accent': pal['accent'],
            'ok':     pal['ok'],
            'warn':   pal['warn'],
            'bad':    pal['bad'],
        }

        CANVAS_BG = self.C['canvas']
        PANEL_BG = bg
        PANEL_FG = self.C['fg']
        self.canvas_bg = CANVAS_BG
        self.panel_bg = PANEL_BG

    def _style_widgets(self):
        """Force the field colours instead of trusting the theme.

        Treeview and Spinbox field backgrounds are the one thing ttk themes
        disagree about across versions - that disagreement is what produced
        white boxes with white text. Setting them explicitly makes the app
        render identically on every machine.
        """
        C = self.C
        try:
            s = self.style if THEME_AVAILABLE else ttk.Style(self)
        except Exception:
            return

        if not THEME_AVAILABLE:
            # No ttkbootstrap: build a dark look on top of 'clam' so the app is
            # still readable rather than falling back to grey system defaults.
            try:
                s.theme_use('clam')
                self.configure(background=C['bg'])
                s.configure('.', background=C['bg'], foreground=C['fg'],
                            fieldbackground=C['field'], bordercolor=C['border'],
                            lightcolor=C['bg'], darkcolor=C['bg'],
                            troughcolor=C['field'], arrowcolor=C['fg'])
                s.configure('TFrame', background=C['bg'])
                s.configure('TLabel', background=C['bg'], foreground=C['fg'])
                s.configure('TLabelframe', background=C['bg'], bordercolor=C['border'])
                s.configure('TLabelframe.Label', background=C['bg'], foreground=C['accent'])
                s.configure('TButton', background=C['head'], foreground=C['fg'])
                s.map('TButton', background=[('active', C['accent'])])
                s.configure('TCheckbutton', background=C['bg'], foreground=C['fg'])
                s.configure('TRadiobutton', background=C['bg'], foreground=C['fg'])
            except Exception as e:
                logger.warning(f"fallback theme: {e}")

        # ---- tables -------------------------------------------------------
        try:
            s.configure('Treeview',
                        background=C['field'], fieldbackground=C['field'],
                        foreground=C['fg'], bordercolor=C['border'],
                        borderwidth=0, rowheight=23)
            s.map('Treeview',
                  background=[('selected', C['accent'])],
                  foreground=[('selected', '#ffffff')])
            s.configure('Treeview.Heading',
                        background=C['head'], foreground=C['fg'],
                        relief='flat', padding=5)
            s.map('Treeview.Heading',
                  background=[('active', C['accent'])],
                  foreground=[('active', '#ffffff')])
        except Exception as e:
            logger.warning(f"treeview style: {e}")

        # ---- entries, spinboxes, comboboxes -------------------------------
        for st in ('TEntry', 'TSpinbox', 'TCombobox'):
            try:
                s.configure(st,
                            fieldbackground=C['field'], background=C['field'],
                            foreground=C['fg'], insertcolor=C['fg'],
                            arrowcolor=C['fg'], bordercolor=C['border'],
                            selectbackground=C['accent'], selectforeground='#ffffff',
                            padding=3)
                s.map(st,
                      fieldbackground=[('readonly', C['field']), ('disabled', C['bg'])],
                      foreground=[('disabled', C['muted'])],
                      arrowcolor=[('active', C['accent'])])
            except Exception as e:
                logger.warning(f"{st} style: {e}")

        # The combobox drop-down is a classic Tk listbox and is only reachable
        # through the option database, never through ttk styles.
        try:
            self.option_add('*TCombobox*Listbox.background', C['field'])
            self.option_add('*TCombobox*Listbox.foreground', C['fg'])
            self.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
            self.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        except Exception:
            pass

        try:
            s.configure('TNotebook', background=C['bg'], borderwidth=0)
            s.configure('TNotebook.Tab', padding=(14, 6))
            s.configure('TLabelframe.Label', foreground=C['accent'])
        except Exception:
            pass

    # ---------------------------------------------------------------------
    def on_closing(self):
        import gc
        try:
            if THEME_AVAILABLE and hasattr(self, 'style'):
                try:
                    self.style = None
                except Exception:
                    pass

            if hasattr(self, 'can'):
                self.can.tk_image = None
                self.can.pil_image = None
            if hasattr(self, 'pcan'):
                self.pcan.tk_image = None
                self.pcan.pil_image = None

            self.pil = None
            self.lbl = None
            self.arr = None
            self.res = None
            self.conf_map = None
            self.anomaly_map = None
            self.i_on = None
            self.i_off = None
            self.proj = {}
            self.seg_data = {}
            self.project_masks = {}
            self.undo_s = []

            gc.collect()
        except Exception:
            pass
        finally:
            self.destroy()

    # ---------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------
    def ui(self):
        top = ttk.Frame(self, padding=5); top.pack(fill=tk.X)
        for t, c, s in [("📂 Open Folder", self.load_dir, "primary"),
                        ("📂 Load Proj", self.load_proj, "primary"),
                        ("💾 Save Proj", self.save_project, "success"),
                        ("📊 Export Excel", self.export_excel, "info"),
                        ("🗺️ Export Maps", self.export_reliability_maps, "warning")]:
            create_button(top, t, c, s).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, textvariable=self.st_var).pack(side=tk.RIGHT, padx=10)

        # Status bar. Packed before the main pane so pack() reserves the bottom
        # edge for it instead of letting the expanding pane swallow the space.
        bar = ttk.Frame(self, padding=(8, 3))
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.file_var = tk.StringVar(value="No file loaded")
        ttk.Label(bar, textvariable=self.file_var,
                  foreground=self.C['muted']).pack(side=tk.LEFT)
        ttk.Label(bar, text="b brush · e eraser · r ROI · f fit · Ctrl+Z undo",
                  foreground=self.C['muted']).pack(side=tk.RIGHT)

        self.main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main = self.main

        # weight=0 on the side panels: any extra window width goes to the SEM
        # view only, instead of being split 1:2:1 and bloating the side panels.
        lf = ttk.PanedWindow(main, orient=tk.VERTICAL)
        main.add(lf, weight=0)

        f1 = ttk.Frame(lf); lf.add(f1, weight=1)
        ttk.Label(f1, text="Explorer", font=('Arial', 9, 'bold')).pack(anchor='w')
        tw = ttk.Frame(f1); tw.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tw, show="tree")
        tsb = ttk.Scrollbar(tw, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_sel_file)

        f2 = ttk.Frame(lf); lf.add(f2, weight=1)
        ttk.Label(f2, text="Spots", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.lst = ThemedListbox(f2); self.lst.pack(fill=tk.BOTH, expand=True)
        self.lst.bind_select(self.on_sel_spot)

        f3 = ttk.Frame(lf); lf.add(f3, weight=1)
        ttk.Label(f3, text="Composition (click an element to include/exclude)",
                  font=('Arial', 9, 'bold')).pack(anchor='w')
        cw = ttk.Frame(f3); cw.pack(fill=tk.BOTH, expand=True)
        self.ctbl = ttk.Treeview(cw, columns=("V"), show="tree headings")
        self.ctbl.heading("#0", text="El"); self.ctbl.column("#0", width=70, stretch=True, anchor="w")
        self.ctbl.heading("V", text="At%"); self.ctbl.column("V", width=60, stretch=False, anchor="e")
        csb = ttk.Scrollbar(cw, orient="vertical", command=self.ctbl.yview)
        self.ctbl.configure(yscrollcommand=csb.set)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ctbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ctbl.bind("<Button-1>", self.on_tog_comp)

        f4 = ttk.Frame(lf); lf.add(f4, weight=1)
        cl = ttk.LabelFrame(f4, text="Class", padding=5); cl.pack(fill=tk.X, pady=5)
        rs = ttk.Frame(cl); rs.pack(fill=tk.X)
        ttk.Label(rs, text="Suggested: ", font=("Arial", 9)).pack(side=tk.LEFT)
        self.lsug = ttk.Label(rs, text="-", font=("Arial", 9, "bold"), foreground="#44ccff")
        self.lsug.pack(side=tk.LEFT)
        self.lrat = ttk.Label(cl, text="Ratio: -"); self.lrat.pack(anchor='w', pady=2)
        ra = ttk.Frame(cl); ra.pack(fill=tk.X)
        ttk.Label(ra, text="Assigned: ").pack(side=tk.LEFT)
        ttk.Combobox(ra, textvariable=self.p_eds, values=self.phases,
                     state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.p_eds.trace_add("write", self.save_cls) if hasattr(self.p_eds, "trace_add") \
            else self.p_eds.trace("w", self.save_cls)

        # ---------------- centre column ----------------
        cf = ttk.PanedWindow(main, orient=tk.VERTICAL); main.add(cf, weight=1)
        sem = ttk.LabelFrame(cf, text="SEM (Interactive)", padding=0); cf.add(sem, weight=3)

        ctrl = ttk.Frame(sem); ctrl.pack(fill=tk.X, pady=2)

        ttk.Label(ctrl, text="B:").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Scale(ctrl, from_=0.5, to=3.0, variable=self.brt, orient=tk.HORIZONTAL,
                  length=110, command=self._on_brt).pack(side=tk.LEFT)
        self.brt_lbl = ttk.Label(ctrl, text="1.0", width=4)
        self.brt_lbl.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(ctrl, text="C:").pack(side=tk.LEFT)
        ttk.Scale(ctrl, from_=0.5, to=3.0, variable=self.cnt, orient=tk.HORIZONTAL,
                  length=110, command=self._on_cnt).pack(side=tk.LEFT)
        self.cnt_lbl = ttk.Label(ctrl, text="1.0", width=4)
        self.cnt_lbl.pack(side=tk.LEFT, padx=2)

        b_fit = create_button(ctrl, "⤢ Fit", self.fit_view, "secondary")
        b_fit.pack(side=tk.LEFT, padx=(10, 0))
        tip(b_fit, "Fit the image to the window (f)")

        mode_f = ttk.Frame(ctrl); mode_f.pack(side=tk.RIGHT, padx=5)
        for m in ["None", "ML Mask", "Raw Brushes", "Reliability", "Anomaly Risk"]:
            ttk.Radiobutton(mode_f, text=m, variable=self.ovl_mode, value=m,
                            command=self.draw).pack(side=tk.LEFT, padx=2)

        self.can = ZoomableImageCanvas(sem, bg_color=self.canvas_bg)
        self.can.pack(fill=tk.BOTH, expand=True)
        self.can.draw = self.draw
        self.can.bind("<ButtonPress-1>", self.ldown)
        self.can.bind("<B1-Motion>", self.ldrag)
        self.can.bind("<ButtonRelease-1>", self.lup)

        pdf = ttk.LabelFrame(cf, text="Map (PDF)"); cf.add(pdf, weight=2)
        self.pcan = ZoomableImageCanvas(pdf, bg_color=self.canvas_bg)
        self.pcan.pack(fill=tk.BOTH, expand=True)
        self.pcan.bind("<ButtonPress-1>", self.pcan.start_pan)
        self.pcan.bind("<B1-Motion>", self.pcan.pan)

        # ---------------- right column ----------------
        rf = ttk.Frame(main); main.add(rf, weight=0)

        st = ttk.LabelFrame(rf, text="Seg Tools", padding=5); st.pack(fill=tk.X, pady=5)
        tr = ttk.Frame(st); tr.pack(fill=tk.X)
        for t, m, s, tt in [("🖌", "brush", "secondary", "Brush - paint the selected phase (b)"),
                            ("⌫", "eraser", "warning", "Eraser - remove labels (e)"),
                            ("⛝", "roi", "info", "Draw region of interest (r)")]:
            btn = create_button(tr, t, lambda m=m: self.tool(m), s, width=3)
            btn.pack(side=tk.LEFT, padx=1)
            tip(btn, tt)
        b_undo = create_button(tr, "⟲", self.undo, "light", width=3)
        b_undo.pack(side=tk.LEFT, padx=1); tip(b_undo, "Undo last brush stroke (Ctrl+Z)")
        b_clr = create_button(tr, "🗑", self.rst, "danger", width=3)
        b_clr.pack(side=tk.RIGHT); tip(b_clr, "Clear all labels, masks and maps for this image")

        sr = ttk.Frame(st); sr.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sr, text="Size:").pack(side=tk.LEFT)
        self._brush_d = tk.DoubleVar(value=float(self.brush_sz.get()))
        ttk.Scale(sr, from_=1, to=30, variable=self._brush_d, orient=tk.HORIZONTAL,
                  length=90, command=self._on_brush).pack(side=tk.LEFT, padx=2)
        self.brush_lbl = ttk.Label(sr, text=str(self.brush_sz.get()), width=3)
        self.brush_lbl.pack(side=tk.LEFT)
        ttk.Label(sr, text="Cut%:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(sr, textvariable=self.cut, from_=0, to=50, increment=1.0,
                    width=5, command=self.upd_roi).pack(side=tk.LEFT)

        ttk.Label(st, text="Phase:").pack(anchor='w', pady=(8, 0))
        ttk.Combobox(st, textvariable=self.p_seg, values=self.phases,
                     state="readonly").pack(fill=tk.X)

        est_frame = ttk.Frame(st); est_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(est_frame, text="RF Trees:").pack(side=tk.LEFT)
        ttk.Spinbox(est_frame, textvariable=self.n_est, from_=10, to=500,
                    increment=10, width=6).pack(side=tk.LEFT, padx=4)

        # Figure geometry for exported maps - must match the MT/Tc app
        pan_frame = ttk.Frame(st); pan_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(pan_frame, text="Fig W (in):").pack(side=tk.LEFT)
        ttk.Spinbox(pan_frame, textvariable=self.fig_w_var, from_=2.0, to=12.0,
                    increment=0.5, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(pan_frame, text="Font (pt):").pack(side=tk.LEFT)
        ttk.Spinbox(pan_frame, textvariable=self.fig_font_var, from_=6.0, to=24.0,
                    increment=1.0, width=5).pack(side=tk.LEFT, padx=2)

        ar = ttk.Frame(st); ar.pack(fill=tk.X, pady=8)
        create_button(ar, "⚡ Auto", self.auto, "info").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        create_button(ar, "▶ Train", self.train, "success").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        create_button(ar, "💾 Mask", self.save_seg, "primary").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        create_button(ar, "🖼️ Pic", self.export_pic, "warning").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        mf = ttk.LabelFrame(rf, text="RF Model Portability", padding=5)
        mf.pack(fill=tk.X, pady=5)
        mr = ttk.Frame(mf); mr.pack(fill=tk.X)
        create_button(mr, "📂 Load Model", self.load_model, "secondary").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        create_button(mr, "💾 Save Model", self.save_model, "secondary").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        create_button(mf, "🎯 Apply Model to Current Image", self.apply_model, "success").pack(fill=tk.X, pady=(5, 0))

        res = ttk.LabelFrame(rf, text="Results", padding=5); res.pack(fill=tk.BOTH, expand=True)
        ex = ttk.Combobox(res, textvariable=self.p_exc, values=["(None)"]+self.phases, state="readonly")
        ex.pack(fill=tk.X)
        ex.bind("<<ComboboxSelected>>", self.calc)

        nb = ttk.Notebook(res); nb.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        tab_leg = ttk.Frame(nb); nb.add(tab_leg, text="Legend")

        # 1. Reliability Scale Bar
        self.cbar_frame = ttk.LabelFrame(tab_leg, text="Reliability Scale")
        self.cbar_frame.pack(fill=tk.X, padx=5, pady=5)
        self.cbar_canvas = tk.Canvas(self.cbar_frame, height=18, bg=self.panel_bg,
                                     highlightthickness=0)
        self.cbar_canvas.pack(fill=tk.X, padx=5, pady=2)
        cbar_lbl = ttk.Frame(self.cbar_frame); cbar_lbl.pack(fill=tk.X, padx=5)
        ttk.Label(cbar_lbl, text="Low (Red)").pack(side=tk.LEFT)
        ttk.Label(cbar_lbl, text="High (Yellow)").pack(side=tk.RIGHT)
        self.cbar_canvas.bind("<Configure>", self.draw_colorbar)

        # 2. Anomaly Risk Scale Bar
        self.abar_frame = ttk.LabelFrame(tab_leg, text="Anomaly Risk Scale")
        self.abar_frame.pack(fill=tk.X, padx=5, pady=5)
        self.abar_canvas = tk.Canvas(self.abar_frame, height=18, bg=self.panel_bg,
                                     highlightthickness=0)
        self.abar_canvas.pack(fill=tk.X, padx=5, pady=2)
        abar_lbl = ttk.Frame(self.abar_frame); abar_lbl.pack(fill=tk.X, padx=5)
        ttk.Label(abar_lbl, text="Low (Clear)").pack(side=tk.LEFT)
        ttk.Label(abar_lbl, text="High (Magenta)").pack(side=tk.RIGHT)
        self.abar_canvas.bind("<Configure>", self.draw_anomaly_colorbar)

        sf = ScrolledFrame(tab_leg); sf.pack(fill=tk.BOTH, expand=True)
        vp = sf.view_port if hasattr(sf, 'view_port') else sf

        hdr = ttk.Frame(vp); hdr.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(hdr, text="Show", foreground=self.C['muted']).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Auto", foreground=self.C['muted']).pack(side=tk.RIGHT)
        create_button(hdr, "none", lambda: self._set_all_vis(False), "link").pack(side=tk.RIGHT, padx=2)
        create_button(hdr, "all", lambda: self._set_all_vis(True), "link").pack(side=tk.RIGHT)

        for i, p in enumerate(self.phases):
            r = ttk.Frame(vp); r.pack(fill=tk.X, pady=1)
            c = '#%02x%02x%02x' % self.colors[i % len(self.colors)][:3]
            create_check(r, self.p_vis[p], self.draw, "primary").pack(side=tk.LEFT)
            ttk.Label(r, text="██", foreground=c).pack(side=tk.LEFT, padx=(2, 0))
            ttk.Label(r, text=p).pack(side=tk.LEFT, padx=6)
            create_check(r, self.p_auto[p], None, "warning").pack(side=tk.RIGHT)

        f_val = ttk.Frame(nb); nb.add(f_val, text="Analysis")
        self.tval = ttk.Treeview(f_val, columns=("C", "A"), show="tree headings")
        self.tval.heading("#0", text="Phase/Type"); self.tval.column("#0", width=95, stretch=False)
        self.tval.heading("C", text="Composition (At%)"); self.tval.column("C", width=145, stretch=True)
        self.tval.heading("A", text="Area%"); self.tval.column("A", width=48, stretch=False, anchor="e")
        vs = ttk.Scrollbar(f_val, orient="vertical", command=self.tval.yview)
        self.tval.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.tval.pack(fill=tk.BOTH, expand=True)

        # Keyboard shortcuts (ignored while typing in an entry or spinbox).
        self.bind("<Key>", self._hotkey)
        self.bind("<Control-z>", lambda e: self.undo())

        # Place the sashes once the window has actually been mapped.
        self.after(120, self._init_sashes)

    # ---------------------------------------------------------------------
    def _set_all_vis(self, state):
        for v in self.p_vis.values():
            v.set(state)
        self.draw()

    def _hotkey(self, e):
        try:
            w = self.focus_get()
            if w is not None and w.winfo_class() in (
                    "TEntry", "Entry", "TCombobox", "TSpinbox", "Spinbox", "Text"):
                return
        except Exception:
            return
        k = e.keysym.lower()
        if k == 'b': self.tool('brush')
        elif k == 'e': self.tool('eraser')
        elif k == 'r': self.tool('roi')
        elif k == 'f': self.fit_view()

    def _fit(self, canvas, img, redraw):
        if img is None: return
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 20 or ch < 20:            # not mapped yet, try again shortly
            self.after(80, lambda: self._fit(canvas, img, redraw))
            return
        s = min(cw / img.width, ch / img.height) * 0.98
        canvas.scale = max(0.02, s)
        canvas.pan_x = int((cw - img.width * canvas.scale) / 2)
        canvas.pan_y = int((ch - img.height * canvas.scale) / 2)
        redraw()

    def fit_view(self, e=None):
        """Scale the SEM image to fit the pane and centre it."""
        self._fit(self.can, self.pil, self.draw)

    def fit_pdf(self):
        self._fit(self.pcan, self.pcan.pil_image, self.pcan.draw)

    def _init_sashes(self):
        """Pin the side panels to sensible widths. Because their weight is 0,
        they keep these widths when the window is resized or maximised."""
        try:
            self.update_idletasks()
            total = self.main.winfo_width() or self._win_w
            left = max(260, min(340, int(total * 0.16)))
            right = max(320, min(420, int(total * 0.20)))
            self.main.sashpos(0, left)
            self.main.sashpos(1, max(left + 320, total - right))
        except Exception as e:
            logger.warning(f"sash init: {e}")

    # ---------------------------------------------------------------------
    # Slider callbacks (ttk.Scale has no 'resolution', so values are
    # formatted for display here instead)
    # ---------------------------------------------------------------------
    def _on_brt(self, v):
        self.brt_lbl.config(text=f"{float(v):.1f}")
        self.draw()

    def _on_cnt(self, v):
        self.cnt_lbl.config(text=f"{float(v):.1f}")
        self.draw()

    def _on_brush(self, v):
        n = max(1, int(round(float(v))))
        self.brush_sz.set(n)
        self.brush_lbl.config(text=str(n))

    # ---------------------------------------------------------------------
    def draw(self, e=None):
        if not self.pil: return
        w, h = int(self.pil.width * self.can.scale), int(self.pil.height * self.can.scale)
        if w < 1: return

        im = ImageEnhance.Contrast(
            ImageEnhance.Brightness(self.pil).enhance(self.brt.get())
        ).enhance(self.cnt.get()).resize((w, h), Image.Resampling.BILINEAR)

        mode = self.ovl_mode.get()

        if mode == "None":
            pass

        elif mode == "Reliability" and self.conf_map is not None:
            c_map = np.array(Image.fromarray((self.conf_map * 255).astype('uint8')).resize((w, h), Image.Resampling.BILINEAR))
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 255
            rgba[:, :, 1] = c_map
            rgba[:, :, 3] = 120
            ov = Image.fromarray(rgba, "RGBA")
            im.paste(ov, (0, 0), ov)

        elif mode == "Anomaly Risk" and self.anomaly_map is not None:
            a_map = np.array(Image.fromarray((self.anomaly_map * 255).astype('uint8')).resize((w, h), Image.Resampling.BILINEAR))
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = a_map     # Red
            rgba[:, :, 1] = 0         # Green
            rgba[:, :, 2] = a_map     # Blue
            rgba[:, :, 3] = np.clip(a_map, 0, 220)   # never a solid wall of colour
            ov = Image.fromarray(rgba, "RGBA")
            im.paste(ov, (0, 0), ov)

        else:
            if mode == "Raw Brushes":
                m = np.array(self.lbl)
            else:
                m = self.res if self.res is not None else np.array(self.lbl)

            if m is not None and m.max() > 0:
                mv = np.array(Image.fromarray(m.astype('uint8')).resize((w, h), Image.Resampling.NEAREST))
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                for i in np.unique(mv):
                    if i == 0: continue
                    i_idx = int(i)
                    pn = self.imap.get(i_idx)
                    if pn and self.p_vis[pn].get():
                        c = list(self.colors[(i_idx - 1) % len(self.colors)])
                        c[3] = 150
                        rgba[mv == i] = tuple(c)
                ov = Image.fromarray(rgba, "RGBA")
                im.paste(ov, (0, 0), ov)

        self.can.tk_image = ImageTk.PhotoImage(im)
        self.can.delete("all")
        self.can.create_image(self.can.pan_x, self.can.pan_y, anchor=tk.NW, image=self.can.tk_image)

        if self.roi:
            s, px, py = self.can.scale, self.can.pan_x, self.can.pan_y
            x1, y1, x2, y2 = self.roi
            self.can.create_rectangle(x1*s+px, y1*s+py, x2*s+px, y2*s+py,
                                      outline="#FFD700", width=2, dash=(4, 4))

        if self.mode in ["brush", "eraser"]:
            mx, my = self.last_m
            r = self.brush_sz.get() * self.can.scale
            self.can.create_oval(mx-r, my-r, mx+r, my+r, outline="cyan", tags="c")

    def draw_colorbar(self, e=None):
        """Red to yellow gradient for the reliability scale bar."""
        w = self.cbar_canvas.winfo_width()
        h = self.cbar_canvas.winfo_height()
        self.cbar_canvas.delete("all")
        if w > 1:
            for i in range(w):
                g = int((i / w) * 255)
                self.cbar_canvas.create_line(i, 0, i, h, fill=f'#ff{g:02x}00')

    def draw_anomaly_colorbar(self, e=None):
        """Black to magenta gradient for the anomaly risk scale bar."""
        w = self.abar_canvas.winfo_width()
        h = self.abar_canvas.winfo_height()
        self.abar_canvas.delete("all")
        if w > 1:
            for i in range(w):
                g = int((i / w) * 255)
                self.abar_canvas.create_line(i, 0, i, h, fill=f'#{g:02x}00{g:02x}')

    # ---------------------------------------------------------------------
    def ldown(self, e):
        if self.drawing: self.rstart = self.can.to_coords(e.x, e.y)
        elif self.mode in ["brush", "eraser"]: self.snap(); self.paint(e)

    def ldrag(self, e):
        self.last_m = (e.x, e.y)
        if self.mode in ["brush", "eraser"]: self.paint(e); self.draw()

    def lup(self, e):
        if self.drawing and self.rstart:
            rx, ry = self.can.to_coords(e.x, e.y)
            x1, x2 = sorted([int(self.rstart[0]), int(rx)])
            y1, y2 = sorted([int(self.rstart[1]), int(ry)])
            self.roi = (max(0, x1), max(0, y1), min(self.pil.width, x2), min(self.pil.height, y2))
            self.drawing = False; self.tool("brush"); self.draw()
        elif self.mode in ["brush", "eraser"]: self.calc()

    def paint(self, e):
        if not self.lbl: return
        if self.res is not None: self.res = None; self.st_var.set("Edited")
        d = ImageDraw.Draw(self.lbl)
        x, y = self.can.to_coords(e.x, e.y)
        r = self.brush_sz.get()/self.can.scale
        v = 0 if self.mode == "eraser" else self.pmap[self.p_seg.get()]
        d.ellipse((x-r, y-r, x+r, y+r), fill=v)

    def tool(self, m):
        self.mode = m
        self.can.config(cursor="tcross" if m == "roi" else "crosshair")
        if m == "roi": self.drawing = True

    def snap(self):
        if len(self.undo_s) > 5: self.undo_s.pop(0)
        self.undo_s.append(self.lbl.copy())

    def undo(self):
        if self.undo_s: self.lbl = self.undo_s.pop(); self.res = None; self.draw(); self.calc()

    def rst(self):
        self.snap()
        self.lbl = Image.new("L", self.pil.size, 0)
        self.res = None
        self.conf_map = None
        self.anomaly_map = None
        if ML_AVAILABLE and hasattr(self, 'ml'):
            self.ml.ok = False
            self.ml.iso_ok = False
        self.draw()
        self.calc()

    def upd_roi(self):
        if not self.pil: return
        h, w = self.arr.shape[:2]
        self.roi = (0, 0, w, int(h*(1.0-self.cut.get()/100.0)))
        self.draw()

    # ---------------------------------------------------------------------
    def auto(self):
        if not ML_AVAILABLE: return
        threading.Thread(target=self._ag, daemon=True).start()

    def _ag(self):
        ph = []
        for p, v in self.p_auto.items():
            if v.get():
                z = 0
                if self.cur_file:
                    s = [d['composition'] for d in self.proj[self.cur_file].values() if d.get('phase') == p]
                    if s:
                        el = set().union(*[c.keys() for c in s])
                        z = get_z({e: sum(c.get(e, 0) for c in s)/len(s) for e in el})
                ph.append({'name': p, 'z': z if z > 0 else est_z(p)})
        if not ph: return

        self.st_var.set("Running Fast Otsu...")

        ph.sort(key=lambda x: x['z'])
        num_classes = len(ph)

        h, w = self.arr.shape[:2]
        x1, y1, x2, y2 = self.roi if self.roi else (0, 0, w, h)
        g = color.rgb2gray(self.arr) if len(self.arr.shape) == 3 else self.arr/255.0
        r = g[y1:y2, x1:x2]

        full_conf = np.zeros((h, w), dtype=float)
        nl = np.zeros(r.shape, dtype=np.uint8)

        if num_classes > 1:
            try:
                thresholds = threshold_multiotsu(r, classes=num_classes)
                regions = np.digitize(r, bins=thresholds)

                for i in range(num_classes):
                    nl[regions == i] = self.pmap[ph[i]['name']]

                roi_conf = np.ones_like(r, dtype=float)
                bounds = [0.0] + list(thresholds) + [1.0]

                for i in range(num_classes):
                    mask = (regions == i)
                    lower_bound = bounds[i]
                    upper_bound = bounds[i+1]
                    midpoint = (lower_bound + upper_bound) / 2.0
                    max_dist = (upper_bound - lower_bound) / 2.0
                    dist_to_mid = np.abs(r[mask] - midpoint)
                    roi_conf[mask] = 1.0 - (dist_to_mid / (max_dist + 1e-6))

                roi_conf = np.clip(roi_conf, 0.0, 1.0)
                full_conf[y1:y2, x1:x2] = roi_conf

            except Exception as e:
                logger.error(f"Otsu failed: {e}")
        else:
            nl[:, :] = self.pmap[ph[0]['name']]
            full_conf[y1:y2, x1:x2] = 1.0

        self.snap()
        self.lbl = Image.new("L", self.pil.size, 0)
        ma = np.array(self.lbl)
        ma[y1:y2, x1:x2] = nl
        self.lbl = Image.fromarray(ma)

        self.res = None
        self.conf_map = full_conf

        self.after(0, lambda: self.st_var.set("Ready"))
        self.after(0, self.draw)
        self.after(0, self.calc)

    def train(self):
        if not ML_AVAILABLE: return
        self.st_var.set("Training...")
        threading.Thread(target=self._tr, daemon=True).start()

    def _tr(self):
        m = np.array(self.lbl)
        if m.max() == 0: return
        r = np.zeros(m.shape, dtype=bool)
        x1, y1, x2, y2 = self.roi if self.roi else (0, 0, m.shape[1], m.shape[0])
        r[y1:y2, x1:x2] = True
        try:
            self.ml.train(self.arr, m, r, n_est=self.n_est.get())
            self.res, self.conf_map, self.anomaly_map = self.ml.predict_all(self.arr, r)

            self.after(0, self.calc); self.after(0, self.draw)
            self.after(0, lambda: self.st_var.set("Done"))
        except Exception as e:
            logger.error(f"Train failed: {e}")
            self.after(0, lambda: self.st_var.set("Error"))

    # ---------------------------------------------------------------------
    def calc(self, e=None):
        if self.lbl is None: return
        m = self.res if self.res is not None else np.array(self.lbl)
        eid = self.pmap.get(self.p_exc.get(), -1)
        u, c = np.unique(m, return_counts=True)
        tot = c[(u != 0) & (u != eid)].sum()

        p_stats = {}
        if tot > 0:
            for i, n in zip(u, c):
                if i == 0 or i == eid: continue
                p_stats[self.imap[i]] = n/tot
        self.seg_data[self.cur_file] = {"stats": p_stats}

        p_data, meas_area = {}, None
        if self.cur_file:
            for s, d in self.proj[self.cur_file].items():
                norm = {}
                cmp, ex = d['composition'], d['excluded_elements']
                vsum = sum(v for k, v in cmp.items() if k not in ex)
                if vsum > 0: norm = {k: v/vsum*100 for k, v in cmp.items() if k not in ex}

                is_ref = any(x in s.lower() for x in ["area", "map", "sum"])
                if is_ref:
                    if norm: meas_area = (s, norm)
                    continue

                ph = d.get('phase')
                if ph and ph in self.phases and norm:
                    p_data.setdefault(ph, []).append((s, norm))

        sample_folder = os.path.dirname(self.cur_file) if self.cur_file else ""
        sample_pool_data = {}
        for fpath, data in self.proj.items():
            if os.path.dirname(fpath) == sample_folder:
                for s, d in data.items():
                    ph = d.get('phase')
                    if ph and ph in self.phases:
                        cmp, ex = d['composition'], d['excluded_elements']
                        vsum = sum(v for k, v in cmp.items() if k not in ex)
                        if vsum > 0:
                            norm = {k: v/vsum*100 for k, v in cmp.items() if k not in ex}
                            sample_pool_data.setdefault(ph, []).append(norm)

        for i in self.tval.get_children(): self.tval.delete(i)

        calc_glob = {}

        for p in self.phases:
            area_pct = p_stats.get(p, 0.0)
            spots = p_data.get(p, [])

            avg_comp = {}
            borrowed = False

            if spots:
                els = set().union(*[x[1].keys() for x in spots])
                for el in els:
                    avg_comp[el] = sum(x[1].get(el, 0) for x in spots) / len(spots)
            elif area_pct > 0 and p in sample_pool_data:
                borrowed = True
                pool_spots = sample_pool_data[p]
                els = set().union(*[x.keys() for x in pool_spots])
                for el in els:
                    avg_comp[el] = sum(x.get(el, 0) for x in pool_spots) / len(pool_spots)

            avg_str = "None"
            if avg_comp:
                avg_str = " ".join([f"{k}:{v:.1f}" for k, v in sorted(avg_comp.items(), key=lambda x: x[1], reverse=True)[:4]])

            if area_pct > 0 or spots:
                display_name = f"{p} (Borrowed)" if borrowed else p
                pid = self.tval.insert("", "end", text=display_name,
                                       values=(avg_str, f"{area_pct*100:.1f}%"), open=True)

                for sname, scomp in spots:
                    cstr = " ".join([f"{k}:{v:.1f}" for k, v in sorted(scomp.items(), key=lambda x: x[1], reverse=True)[:4]])
                    self.tval.insert(pid, "end", text=sname, values=(cstr, " "))
                if len(spots) > 0:
                    self.tval.insert(pid, "end", text="Average", values=(avg_str, " "), tags=('avg',))
                elif borrowed:
                    self.tval.insert(pid, "end", text="From Sample Pool", values=(avg_str, " "), tags=('dev',))

            if area_pct > 0 and avg_comp:
                for el, val in avg_comp.items():
                    calc_glob[el] = calc_glob.get(el, 0) + val * area_pct

        self.tval.insert("", "end", text="---", values=("---", "---"))
        glob_str = "-"
        if calc_glob:
            glob_str = " ".join([f"{k}:{v:.1f}" for k, v in sorted(calc_glob.items(), key=lambda x: x[1], reverse=True)[:5]])
        self.tval.insert("", "end", text="Calc. Global", values=(glob_str, "100%"), tags=('bold',))

        if meas_area:
            n, rc = meas_area
            mstr = " ".join([f"{k}:{v:.1f}" for k, v in sorted(rc.items(), key=lambda x: x[1], reverse=True)[:5]])
            self.tval.insert("", "end", text=f"Meas. ({n})", values=(mstr, "Ref"), tags=('ref',))

            if calc_glob:
                diffs = {}
                all_els = set(calc_glob.keys()) | set(rc.keys())
                for el in all_els:
                    d_val = calc_glob.get(el, 0.0) - rc.get(el, 0.0)
                    if abs(d_val) > 0.1:
                        diffs[el] = d_val

                if diffs:
                    dstr = " ".join([f"{k}:{v:+.1f}" for k, v in sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]])
                    self.tval.insert("", "end", text="Deviation", values=(dstr, "Delta"), tags=('dev',))

        self.tval.tag_configure('bold', font=('Arial', 9, 'bold'))
        self.tval.tag_configure('ref', foreground='cyan')
        self.tval.tag_configure('avg', font=('Arial', 9, 'italic'))
        self.tval.tag_configure('dev', foreground='orange')

    # ---------------------------------------------------------------------
    def export_reliability_maps(self):
        if self.cur_file and self.lbl:
            self.save_seg()

        if not self.project_masks:
            messagebox.showwarning("No Data", "Please load a project or train a mask first.")
            return

        out_dir = filedialog.askdirectory(title="Select Folder to Save Reliability Maps")
        if not out_dir: return

        self.st_var.set("Generating Maps...")
        threading.Thread(target=self._generate_maps_thread, args=(out_dir,), daemon=True).start()

    def _generate_maps_thread(self, out_dir):
        if not ML_AVAILABLE:
            self.after(0, lambda: self.st_var.set("ML Libraries missing"))
            return

        saved_count = 0
        cut_pct = self.cut.get() / 100.0

        try:
            fig_w = float(self.fig_w_var.get())
            fs = float(self.fig_font_var.get())
        except Exception:
            fig_w, fs = PANEL_FIG_W, PANEL_FONT
        if fig_w < 2.0: fig_w = 2.0
        if fs < 6.0: fs = 6.0

        for excel_path, mask_data in self.project_masks.items():
            base_name = os.path.splitext(excel_path)[0]
            img_path = None
            for ext in [".tif", ".TIF", ".tiff", ".TIFF", ".png", ".jpg"]:
                if os.path.exists(base_name + ext):
                    img_path = base_name + ext
                    break

            if not img_path: continue

            if isinstance(mask_data, dict) and "lbl" in mask_data:
                actual_mask = mask_data["lbl"]
            else:
                actual_mask = mask_data

            if actual_mask is None or np.max(actual_mask) == 0:
                continue

            try:
                img_pil = load_sem_image_robust(img_path)
                if img_pil is None: continue

                img_base = ImageEnhance.Contrast(
                    ImageEnhance.Brightness(img_pil).enhance(self.brt.get())
                ).enhance(self.cnt.get())
                img_arr = np.array(img_base)

                h, w = img_arr.shape[:2]
                y_max = int(h * (1.0 - cut_pct))

                roi_mask = np.zeros((h, w), dtype=bool)
                roi_mask[0:y_max, 0:w] = True

                temp_ml = Segmenter()
                temp_ml.train(img_arr, actual_mask, roi=roi_mask, n_est=self.n_est.get())

                # One feature pass for all three exports.
                res_mask, reliability, anomaly_map = temp_ml.predict_all(img_arr, roi=roi_mask)

                # 1. Segmented mask overlay
                if res_mask is not None and res_mask.max() > 0:
                    im_mask = img_base.copy()
                    mv = np.array(Image.fromarray(res_mask.astype('uint8')).resize((w, h), Image.Resampling.NEAREST))
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    for i in np.unique(mv):
                        if i == 0: continue
                        i_idx = int(i)
                        pn = self.imap.get(i_idx)
                        if pn and self.p_vis[pn].get():
                            c = list(self.colors[(i_idx - 1) % len(self.colors)])
                            c[3] = 150
                            rgba[mv == i] = tuple(c)

                    ov_mask = Image.fromarray(rgba, "RGBA")
                    im_mask.paste(ov_mask, (0, 0), ov_mask)
                    mask_save_path = os.path.join(out_dir, f"{os.path.basename(base_name)}_Segmented.png")
                    im_mask.convert("RGB").save(mask_save_path)

                # 2. Reliability map
                reliability_cropped = reliability[0:y_max, 0:w]

                fig = make_map_figure(reliability_cropped, 'magma',
                                      'Classification reliability', y_max, w,
                                      fig_w=fig_w, fs=fs)
                rel_save_path = os.path.join(out_dir, f"{os.path.basename(base_name)}_ReliabilityMap.png")
                # No bbox_inches='tight': the file keeps its exact physical size,
                # so the font is true when placed in the panel at 100% scale.
                fig.savefig(rel_save_path, format='png', dpi=PANEL_DPI)
                fig.savefig(rel_save_path.replace('.png', '.pdf'), format='pdf')

                # 3. Anomaly risk map
                if anomaly_map is not None:
                    anomaly_cropped = anomaly_map[0:y_max, 0:w]
                    magenta_cmap = LinearSegmentedColormap.from_list('magenta', ['#000000', '#FF00FF'])

                    fig_a = make_map_figure(anomaly_cropped, magenta_cmap,
                                            'Anomaly risk', y_max, w,
                                            fig_w=fig_w, fs=fs)
                    anom_save_path = os.path.join(out_dir, f"{os.path.basename(base_name)}_AnomalyMap.png")
                    fig_a.savefig(anom_save_path, format='png', dpi=PANEL_DPI)
                    fig_a.savefig(anom_save_path.replace('.png', '.pdf'), format='pdf')

                saved_count += 1
                self.after(0, lambda n=saved_count: self.st_var.set(f"Exported {n} image sets..."))

            except Exception as e:
                logger.error(f"Failed map for {base_name}: {e}")
                continue

        self.after(0, lambda: self.st_var.set("Ready"))
        self.after(0, lambda: messagebox.showinfo(
            "Success",
            f"Exported {saved_count} full image sets (Segments, Reliability, and Anomalies) to:\n{out_dir}"))

    # ---------------------------------------------------------------------
    def load_model(self):
        if not ML_AVAILABLE: return
        f = filedialog.askopenfilename(title="Select Trained RF Model",
                                       filetypes=[("Joblib Model", "*.joblib")])
        if not f: return
        try:
            obj = joblib.load(f)
            has_anom = self.ml.load_bundle(obj)

            saved_phases = obj.get("phases") if isinstance(obj, dict) else None

            msg = "Model loaded. You can now use 'Apply Model' on new images.\n\n"
            msg += ("Anomaly detector: included.\n" if has_anom else
                    "Anomaly detector: NOT in this file.\n"
                    "It was saved before anomaly risk existed, so risk maps are\n"
                    "unavailable until you retrain and save the model again.\n")

            if saved_phases and list(saved_phases) != list(self.phases):
                msg += ("\nWarning: this model was trained on a different phase list:\n"
                        f"  saved:   {', '.join(saved_phases)}\n"
                        f"  current: {', '.join(self.phases)}\n"
                        "Label numbers may map to the wrong phases.")

            self.st_var.set("Model loaded" + ("" if has_anom else " (no anomaly detector)"))
            messagebox.showinfo("Model loaded", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def save_model(self):
        if not ML_AVAILABLE or not self.ml.ok:
            messagebox.showwarning("No Model", "There is no trained model in memory. Please use 'Train' first.")
            return
        f = filedialog.asksaveasfilename(title="Save Trained RF Model", defaultextension=".joblib",
                                         filetypes=[("Joblib Model", "*.joblib")])
        if not f: return
        try:
            # Save the isolation forest alongside the random forest, plus the
            # phase list so a mismatch can be flagged on load.
            joblib.dump(self.ml.bundle(phases=list(self.phases),
                                       pmap=dict(self.pmap)), f)
            extra = "" if self.ml.iso_ok else \
                "\n\nNote: no anomaly detector was included, because the model " \
                "in memory came from an older file. Retrain to add one."
            messagebox.showinfo("Success", f"Trained model saved to:\n{f}{extra}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model:\n{e}")

    def apply_model(self):
        if not ML_AVAILABLE or not self.ml.ok:
            messagebox.showwarning("No Model", "No model is loaded or trained.")
            return
        if self.arr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self.st_var.set("Applying Loaded Model...")

        def _apply():
            try:
                h, w = self.arr.shape[:2]
                r = np.ones((h, w), dtype=bool)
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    r = np.zeros((h, w), dtype=bool)
                    r[y1:y2, x1:x2] = True

                # All three maps from one feature pass. The anomaly map used to
                # be skipped here entirely, which is why risk never appeared on
                # images segmented with a loaded model.
                self.res, self.conf_map, self.anomaly_map = self.ml.predict_all(self.arr, r)

                self.after(0, self.calc)
                self.after(0, self.draw)
                if self.ml.iso_ok:
                    self.after(0, lambda: self.st_var.set("Model applied"))
                else:
                    self.after(0, lambda: self.st_var.set("Model applied (no anomaly detector)"))
            except Exception as e:
                self.after(0, lambda: self.st_var.set("Apply Error"))
                logger.error(f"Failed to apply model: {e}")

        threading.Thread(target=_apply, daemon=True).start()

    def save_seg(self):
        if self.cur_file:
            saved_data = {
                "lbl": np.array(self.lbl).astype('uint8'),
                "ml": self.res.astype('uint8') if self.res is not None else None
            }
            self.project_masks[self.cur_file] = saved_data

    def export_pic(self):
        if not self.pil:
            messagebox.showwarning("No Image", "There is no image currently loaded to save.")
            return

        f = filedialog.asksaveasfilename(title="Save Image Layers Prefix", defaultextension=".png",
                                         filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if not f: return

        base_path, ext = os.path.splitext(f)

        im_base = ImageEnhance.Contrast(
            ImageEnhance.Brightness(self.pil).enhance(self.brt.get())
        ).enhance(self.cnt.get())
        w, h = im_base.width, im_base.height

        try:
            # 1. Raw enhanced image
            im_raw = im_base.convert("RGB") if ext.lower() in [".jpg", ".jpeg"] else im_base
            im_raw.save(f"{base_path}_raw{ext}")

            # 2. Segmentation mask overlay
            im_mask = im_base.copy()
            m = self.res if self.res is not None else np.array(self.lbl)
            if m is not None and m.max() > 0:
                mv = np.array(Image.fromarray(m.astype('uint8')).resize((w, h), Image.Resampling.NEAREST))
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                for i in np.unique(mv):
                    if i == 0: continue
                    i_idx = int(i)
                    pn = self.imap.get(i_idx)
                    if pn and self.p_vis[pn].get():
                        c = list(self.colors[(i_idx - 1) % len(self.colors)])
                        c[3] = 150
                        rgba[mv == i] = tuple(c)

                ov_mask = Image.fromarray(rgba, "RGBA")
                im_mask.paste(ov_mask, (0, 0), ov_mask)

            im_mask_final = im_mask.convert("RGB") if ext.lower() in [".jpg", ".jpeg"] else im_mask
            im_mask_final.save(f"{base_path}_mask{ext}")

            # 3. Reliability overlay
            if self.conf_map is not None:
                im_rel = im_base.copy()
                c_map = np.array(Image.fromarray((self.conf_map * 255).astype('uint8')).resize((w, h), Image.Resampling.BILINEAR))
                rgba_rel = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_rel[:, :, 0] = 255
                rgba_rel[:, :, 1] = c_map
                rgba_rel[:, :, 3] = 120
                ov_rel = Image.fromarray(rgba_rel, "RGBA")
                im_rel.paste(ov_rel, (0, 0), ov_rel)

                im_rel_final = im_rel.convert("RGB") if ext.lower() in [".jpg", ".jpeg"] else im_rel
                im_rel_final.save(f"{base_path}_reliability{ext}")

            # 4. Anomaly risk overlay
            if self.anomaly_map is not None:
                im_anom = im_base.copy()
                a_map = np.array(Image.fromarray((self.anomaly_map * 255).astype('uint8')).resize((w, h), Image.Resampling.BILINEAR))
                rgba_anom = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_anom[:, :, 0] = a_map
                rgba_anom[:, :, 1] = 0
                rgba_anom[:, :, 2] = a_map
                rgba_anom[:, :, 3] = np.clip(a_map, 0, 220)

                ov_anom = Image.fromarray(rgba_anom, "RGBA")
                im_anom.paste(ov_anom, (0, 0), ov_anom)

                im_anom_final = im_anom.convert("RGB") if ext.lower() in [".jpg", ".jpeg"] else im_anom
                im_anom_final.save(f"{base_path}_anomaly{ext}")

            messagebox.showinfo("Success", f"Exported all layers to:\n{base_path}_...{ext}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images:\n{e}")

    # ---------------------------------------------------------------------
    def load_dir(self):
        d = filedialog.askdirectory()
        if not d: return
        self.tree.delete(*self.tree.get_children())
        self.proj, self.seg_data, self.project_masks = {}, {}, {}
        self.st_var.set("Scanning...")
        self.update_idletasks()
        fm = {d: ""}
        for r, _, f in os.walk(d):
            pid = fm.get(os.path.dirname(r), "")
            if r != d: fm[r] = self.tree.insert(pid, "end", text=os.path.basename(r), open=False)
            for x in sorted(f):
                if x.lower().endswith(('.xlsx', '.xls')) and not x.startswith("~$"):
                    self.tree.insert(fm[r], "end", text=x, values=["file"], tags=(os.path.join(r, x),))
        self.st_var.set("Ready")

    def on_sel_file(self, e):
        if self.cur_file and self.lbl: self.save_seg()

        sel = self.tree.selection()
        if not sel: return
        path = self.tree.item(sel[0])['tags'][0] if self.tree.item(sel[0])['tags'] else None
        if not path: return
        self.cur_file = path

        if path not in self.proj:
            d, err = parse_eds(path)
            if err:
                messagebox.showerror("Excel import failed", f"{os.path.basename(path)}\n\n{err}")
                self.st_var.set("Excel import failed")
                return

            self.proj[path] = d
            for s, dat in self.proj[path].items():
                if not dat.get('phase'):
                    if any(x in s.lower() for x in ["selected area", "map", "sum"]):
                        dat['phase'] = "Area"
                    else:
                        c = dat['composition']; vs = sum(c.values())
                        nc = {k: v/vs*100 for k, v in c.items()} if vs > 0 else c
                        sug = suggest(nc, self.rules)
                        if sug: dat['phase'] = sug

        self.lst.delete(0, tk.END)
        active = set()
        for s, d in self.proj[path].items():
            self.lst.insert(tk.END, s)
            if d.get('phase'): active.add(d['phase'])
            sug = suggest(d['composition'], self.rules)
            if sug: active.add(sug)

        for p in self.phases:
            if p != "Area":
                self.p_auto[p].set(p in active)
            else:
                self.p_auto[p].set(False)

        b = os.path.splitext(path)[0]
        self.img_path, self.pil = None, None
        for x in [".tif", ".TIF", ".tiff", ".TIFF"]:
            if os.path.exists(b+x):
                self.img_path = b+x; self.pil = load_sem_image_robust(self.img_path)
                self.arr = np.array(self.pil)

                if path in self.project_masks:
                    saved = self.project_masks[path]
                    if isinstance(saved, dict) and "lbl" in saved:
                        self.lbl = Image.fromarray(saved["lbl"])
                        self.res = saved["ml"]
                    else:
                        self.lbl = Image.fromarray(saved)
                        self.res = None
                    self.st_var.set("Masks loaded from Project")
                else:
                    self.lbl = Image.new("L", self.pil.size, 0)
                    self.res = None

                self.conf_map = None
                self.anomaly_map = None
                self.can.set_image(self.pil); self.upd_roi()
                self.after(60, self.fit_view)
                break

        n_spots = self.lst.size()
        self.file_var.set(
            f"{os.path.basename(path)}   ·   {n_spots} spot(s)"
            + ("   ·   no matching image found" if self.pil is None else ""))

        self.pcan.delete("all")
        if os.path.exists(b+".pdf") and fitz is not None:
            try:
                doc = fitz.open(b+".pdf")
                pix = doc[0].get_pixmap(dpi=150)
                self.pcan.set_image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                self.after(60, self.fit_pdf)
            except Exception:
                pass
        if self.lst.size() > 0: self.lst.select_set(0); self.on_sel_spot(None)
        self.calc()

    def on_sel_spot(self, e):
        sel = self.lst.curselection()
        if not sel: return
        self.cur_spot = self.lst.get(sel[0])
        if not self.cur_spot: return
        d = self.proj[self.cur_file][self.cur_spot]
        for i in self.ctbl.get_children(): self.ctbl.delete(i)
        c = d['composition']

        ex_elems = d.get('excluded_elements', [])
        vsum = sum(v for k, v in c.items() if k not in ex_elems)
        for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
            ex = k in ex_elems
            val = f"{v/vsum*100:.1f}" if (vsum > 0 and not ex) else f"({v})"
            icon = self.i_off if ex else self.i_on
            self.ctbl.insert("", "end", text=k, values=(val,), image=icon, tags=('ex',) if ex else ())
        self.ctbl.tag_configure('ex', foreground='gray')

        norm_c = {k: v/vsum*100 for k, v in c.items() if k not in ex_elems} if vsum else {}
        is_area = any(x in self.cur_spot.lower() for x in ["selected area", "map", "sum"])
        sug = "Area" if is_area else suggest(norm_c, self.rules)
        self.lsug.config(text=sug if sug else "None")
        self.p_eds.set(d['phase'] if d['phase'] else sug)

        rs = sum(norm_c.get(e, 0) for e in RE_ELEMENTS)
        ts = sum(norm_c.get(e, 0) for e in TM_ELEMENTS)
        self.lrat.config(text=f"TM/RE: {ts/rs:.2f}" if rs > 0 else "N/A")

    def on_tog_comp(self, e):
        r = self.ctbl.identify_row(e.y); c = self.ctbl.identify_column(e.x)
        if not r or c != "#0": return
        el = self.ctbl.item(r, "text")

        ex = self.proj[self.cur_file][self.cur_spot].setdefault('excluded_elements', [])
        if el in ex: ex.remove(el)
        else: ex.append(el)
        self.on_sel_spot(None)
        self.calc()

    def save_cls(self, *a):
        if self.cur_file and self.cur_spot:
            self.proj[self.cur_file][self.cur_spot]['phase'] = self.p_eds.get()

    def tog_exc(self):
        sel = self.ctbl.selection()
        if not sel: return
        el = self.ctbl.item(sel[0])['values'][0]
        ex = self.proj[self.cur_file][self.cur_spot]['excluded_elements']
        if el in ex: ex.remove(el)
        else: ex.append(el)
        self.on_sel_spot(None)

    # ---------------------------------------------------------------------
    def save_project(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f:
            if self.cur_file and self.lbl: self.save_seg()

            base_dir = os.path.dirname(f)

            def make_rel(path_dict):
                rel_dict = {}
                for path, data in path_dict.items():
                    try:
                        rel_path = os.path.relpath(path, base_dir)
                    except ValueError:
                        rel_path = path
                    rel_dict[rel_path] = data
                return rel_dict

            rel_proj = make_rel(self.proj)
            rel_seg = make_rel(self.seg_data)

            json.dump({"eds_data": rel_proj, "segmentation": rel_seg}, open(f, "w"))

            npz_file = f.replace(".json", "_masks.npz")
            if self.project_masks:
                rel_masks = make_rel(self.project_masks)
                np.savez_compressed(npz_file, data=np.array([rel_masks], dtype=object))
            messagebox.showinfo("Success", "Project Database & Masks Saved Portably!")

    def load_proj(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            try:
                base_dir = os.path.dirname(f)
                d = json.load(open(f))

                raw_proj = d.get("eds_data", {})
                raw_seg = d.get("segmentation", {})

                def make_abs(path_dict):
                    abs_dict = {}
                    for old_path, data in path_dict.items():
                        abs_path = os.path.abspath(os.path.join(base_dir, old_path))

                        if not os.path.exists(abs_path):
                            clean_path = old_path.replace('\\', '/')
                            parts = clean_path.split('/')
                            target_file = parts[-1]

                            if len(parts) >= 2:
                                parent_folder = parts[-2]
                                guess = os.path.join(base_dir, parent_folder, target_file)
                                if os.path.exists(guess):
                                    abs_path = guess

                            if not os.path.exists(abs_path):
                                for root, dirs, files in os.walk(base_dir):
                                    if target_file in files:
                                        abs_path = os.path.join(root, target_file)
                                        break

                        abs_dict[abs_path] = data
                    return abs_dict

                self.proj = make_abs(raw_proj)
                self.seg_data = make_abs(raw_seg)

                npz_file = f.replace(".json", "_masks.npz")
                self.project_masks = {}
                if os.path.exists(npz_file):
                    with np.load(npz_file, allow_pickle=True) as loaded:
                        raw_masks = loaded['data'][0] if 'data' in loaded else {k: loaded[k] for k in loaded.files}
                        self.project_masks = make_abs(raw_masks)

                self.tree.delete(*self.tree.get_children())
                fm = {}
                for path in sorted(self.proj.keys()):
                    dn, fn = os.path.dirname(path), os.path.basename(path)
                    if dn not in fm:
                        pid = self.tree.insert("", "end", text=os.path.basename(dn), open=False)
                        fm[dn] = pid
                    self.tree.insert(fm[dn], "end", text=fn, values=["file"], tags=(path,))
                messagebox.showinfo("Success", f"Loaded {len(self.proj)} files")
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {e}")

    def export_excel(self):
        if not self.proj: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not f: return

        rename_map = {"SmFe12": "Sm1Fe12"}

        for folder_node in self.tree.get_children():
            for file_node in self.tree.get_children(folder_node):
                tags = self.tree.item(file_node)['tags']
                if tags:
                    path = tags[0]
                    if path not in self.proj:
                        d, err = parse_eds(path)
                        if not err:
                            self.proj[path] = d
                            for s, dat in d.items():
                                if not dat.get('phase'):
                                    if any(x in s.lower() for x in ["selected area", "map", "sum"]):
                                        dat['phase'] = "Area"
                                    else:
                                        c = dat['composition']; vs = sum(c.values())
                                        nc = {k: v/vs*100 for k, v in c.items()} if vs > 0 else c
                                        sug = suggest(nc, self.rules)
                                        if sug: dat['phase'] = sug

        grouped_data = {}
        for fpath, spots in self.proj.items():
            dirname = os.path.dirname(fpath)
            sample_folder = os.path.basename(dirname)
            if sample_folder not in grouped_data:
                grouped_data[sample_folder] = {"files": [], "area_comps": [], "seg_stats": {}, "seg_img": ""}
            grouped_data[sample_folder]["files"].append((fpath, spots))
            if fpath in self.seg_data:
                grouped_data[sample_folder]["seg_stats"] = self.seg_data[fpath].get("stats", {})

        rows_summary, rows_phase, rows_spots = [], [], []

        for sample_name, data in grouped_data.items():
            for fpath, spots in data["files"]:
                for s, d in spots.items():
                    is_area = d.get('phase') == "Area" or any(x in s.lower() for x in ["selected area", "map", "sum"])
                    if is_area:
                        c = d['composition']; ex = d.get('excluded_elements', [])
                        vsum = sum(v for k, v in c.items() if k not in ex)
                        if vsum > 0:
                            data["area_comps"].append({k: v/vsum*100 for k, v in c.items() if k not in ex})

            avg_area = {}
            if data["area_comps"]:
                all_els = set().union(*[c.keys() for c in data["area_comps"]])
                avg_area = {el: sum(c.get(el, 0) for c in data["area_comps"])/len(data["area_comps"]) for el in all_els}

            for fpath, spots in data["files"]:
                file_name = os.path.basename(fpath)

                raw_seg_stats = self.seg_data.get(fpath, {}).get("stats", {})
                file_seg_stats = {}
                for k, v in raw_seg_stats.items():
                    new_k = rename_map.get(k, k)
                    file_seg_stats[new_k] = file_seg_stats.get(new_k, 0) + v

                ph_acc = {}

                for s, d in spots.items():
                    phase_val = d.get('phase', '')

                    if phase_val in rename_map:
                        phase_val = rename_map[phase_val]

                    if not phase_val:
                        if "selected area" in s.lower():
                            phase_val = "Area"
                        else:
                            c = d['composition']; ex = d.get('excluded_elements', [])
                            vsum = sum(v for k, v in c.items() if k not in ex)
                            nc = {k: v/vsum*100 for k, v in c.items() if k not in ex} if vsum else {}
                            phase_val = suggest(nc, self.rules)

                    c = d['composition']; ex = d.get('excluded_elements', [])
                    vsum = sum(v for k, v in c.items() if k not in ex)
                    norm_c = {}
                    if vsum > 0: norm_c = {el: (val/vsum)*100 for el, val in c.items() if el not in ex}

                    r_spot = {"Sample_Folder": sample_name, "File": file_name, "Spot": s,
                              "Phase": phase_val, "Excluded": ",".join(ex)}
                    r_spot.update(norm_c)
                    for pn, pct in file_seg_stats.items(): r_spot[f"Img_Seg_{pn}%"] = pct * 100
                    rows_spots.append(r_spot)

                    if phase_val != "Area" and not any(x in s.lower() for x in ["area", "map", "sum"]) and norm_c:
                        ph_acc.setdefault(phase_val, []).append(norm_c)

                calc_glob = {}
                for p, comps in ph_acc.items():
                    els = set().union(*[x.keys() for x in comps])
                    p_avg = {e: sum(x.get(e, 0) for x in comps)/len(comps) for e in els}

                    r_ph = {"Sample_Folder": sample_name, "File": file_name, "Phase": p, "Count": len(comps)}
                    r_ph.update(p_avg)
                    rows_phase.append(r_ph)

                    frac = file_seg_stats.get(p, 0)
                    if frac > 0:
                        for e, v in p_avg.items(): calc_glob[e] = calc_glob.get(e, 0) + v * frac

                if calc_glob:
                    r_calc = {"Sample_Folder": sample_name, "File": file_name, "Spot": "Calculated Bulk",
                              "Phase": "Calculated", "Excluded": "None"}
                    r_calc.update(calc_glob)
                    for pn, pct in file_seg_stats.items(): r_calc[f"Img_Seg_{pn}%"] = pct * 100
                    rows_spots.append(r_calc)

                r_sum = {"Sample_Folder": sample_name, "File": file_name, "Type": "File Analysis"}
                for p, v in file_seg_stats.items(): r_sum[f"Seg_{p}%"] = v*100
                for e, v in calc_glob.items(): r_sum[f"Calc_{e}%"] = v
                if calc_glob and avg_area:
                    all_e = set(calc_glob.keys()) | set(avg_area.keys())
                    for e in all_e: r_sum[f"Diff_{e}"] = calc_glob.get(e, 0) - avg_area.get(e, 0)
                rows_summary.append(r_sum)

            ravg = {"Sample_Folder": sample_name, "File": "AVERAGE", "Type": "Sample Measured Area"}
            for e, v in avg_area.items(): ravg[f"Meas_Area_{e}%"] = v
            rows_summary.append(ravg)

        try:
            with pd.ExcelWriter(f) as writer:
                pd.DataFrame(rows_summary).to_excel(writer, sheet_name='Summary', index=False)
                pd.DataFrame(rows_phase).to_excel(writer, sheet_name='Phase_Avgs', index=False)
                pd.DataFrame(rows_spots).to_excel(writer, sheet_name='Spots_Raw', index=False)
            messagebox.showinfo("Success", f"Exported {len(rows_summary)} rows.")
        except Exception as e:
            messagebox.showerror("Export Error", f"{e}")


if __name__ == "__main__":
    try:
        app = EDSBatchAnalyzerApp()
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if 'app' in locals():
                app.on_closing()
        except Exception:
            pass


