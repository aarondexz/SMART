# Module I — SEM & EDS Pipeline

Interactive segmentation of SEM micrographs with chemical phase assignment from EDS point
analyses. Sparse hand-painted annotations train a Random Forest over a texture feature
stack; an Isolation Forest trained on the same pixels flags regions the classifier has
never seen.

## Why not fully automatic

Chemically distinct magnetic phases can be nearly indistinguishable in Z-contrast — in
Sm–Fe–V the 1:12, 3:29 and 2:17 phases differ by roughly 1–2 at.% Sm. Threshold methods
(Otsu, K-means) also read illumination gradients as phases. A few brush strokes from
someone who knows the material resolve both problems, and the trained model then redeploys
across the whole image series unattended.

---

## Input data layout

The app pairs files **by basename** inside each sample folder:

```
MyProject/
└── Sample_A/                 <- folder name becomes the merge key
    ├── image01.xlsx          EDS spot table (required)
    ├── image01.tif           SEM micrograph (required)
    └── image01.pdf           EDS map, optional preview
```

The Excel file must contain a header row with an `Element` column and an `Atomic %`
(or `At%`) column. Spot blocks are found by names like `EDS Spot 1`, `Spectrum 2`,
`Selected Area 1`. Spots whose name contains *area*, *map* or *sum* are treated as the
measured bulk reference rather than as a phase.

Supported micrograph formats: `.tif`, `.tiff` (including 16-bit, auto-rescaled).

---

## Phase rules — `config/phases_config.json`

Phase identity lives in this file, not in the source code. Edit it for your alloy system;
nothing needs recompiling. Rules are evaluated top to bottom and the first full match wins,
so put the most specific phases first.

```json
[
  {
    "name": "Sm1Fe12",
    "rules": [
      {"type": "element", "target": "Sm", "min": 6.0,  "max": 8.5},
      {"type": "ratio", "numerator": ["Fe","V"], "denominator": ["Sm"],
       "min": 10.5, "max": 13.5}
    ]
  },
  {
    "name": "Sm2Fe17",
    "rules": [
      {"type": "element", "target": "Sm", "min": 9.0, "max": 12.0}
    ]
  },
  {"name": "FeV", "rules": [{"type": "element", "target": "Sm", "min": 0.0, "max": 1.5}]}
]
```

Two rule types: `element` bounds an element's at.% directly; `ratio` bounds the sum of the
numerator elements over the sum of the denominator elements. All rules in a phase must hold.

Place this file next to the executable (or the working directory when running from source).
If it is missing the app falls back to a bare `Matrix` / `Precipitate` list.

---

## Workflow

1. **Open Folder** — point at the project root; the explorer lists every `.xlsx` found.
2. Select a file. Spots load on the left with a suggested phase from the JSON rules; click
   an element in the composition table to exclude it from normalisation (useful for carbon
   coating or oxygen).
3. **Paint.** Pick a phase, brush a few strokes on each. `b` brush, `e` eraser, `r` ROI,
   `f` fit view, `Ctrl+Z` undo. Set *Cut%* to exclude the SEM data banner at the bottom.
4. **Train.** The RF and the anomaly detector fit together. Switch overlays between
   *ML Mask*, *Reliability* and *Anomaly Risk*.
5. **Read the anomaly map.** Bright magenta means pixels unlike anything you painted —
   usually an unannotated phase. Paint it, retrain, and residual risk should collapse to
   the phase boundaries only. That is your convergence criterion.
6. **Save Model** to reuse on the rest of the series (*Apply Model to Current Image*).
7. **Export Excel** for the whole project, and **Export Maps** for publication figures.

## Outputs

`Export Excel` writes three sheets:

- `Summary` — one row per micrograph: segmented area fractions, reconstructed bulk
  composition, and its deviation from the measured area reference.
- `Phase_Avgs` — mean composition per phase per micrograph.
- `Spots_Raw` — every EDS spot, normalised, with its phase assignment.

`Sample_Folder` in these sheets is the column that Module II joins against.

`Export Maps` writes, per micrograph, a segmented overlay PNG plus reliability and anomaly
maps as PNG (600 dpi) and PDF, at the exact physical width and font size set in the panel —
matching Module II so both render at identical lettering size in a composite figure.

## Projects and models

**Save Proj** writes a `.json` with all spot classifications plus a `_masks.npz` beside it.
Paths are stored relative to the JSON, so the project folder can be moved or shared.
**Save Model** writes a `.joblib` bundle containing the Random Forest, the Isolation Forest
and the phase list; loading a model trained on a different phase list raises a warning,
because label numbers would otherwise map to the wrong phases.

## Notes and limits

- Segmentation is per-pixel over greyscale texture features, so it will not separate two
  phases that are genuinely identical in both intensity and texture. That is what the EDS
  spots and the thermomagnetic anchor are for.
- Training needs at least 50 labelled pixels; the app will tell you to paint more.
- Training runs on a background thread — the window stays responsive, but wait for the
  status bar to read *Done* before exporting.
