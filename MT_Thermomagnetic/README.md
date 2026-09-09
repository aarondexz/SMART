# Module II — Thermomagnetic Analysis & Database Integration

Curie-temperature extraction from *M*(*T*) curves by automated dual-tangent intersection,
across a whole folder of samples, with publication-ready figure export. Also hosts the
merge step that joins Module I and Module II into one correlative database.

## Why dual-tangent

The inflection of d*M*/d*T* is convenient but shifts with smoothing width and with the
shape of neighbouring transitions. Intersecting a tangent through the steepest part of the
transition with the baseline above it gives a construction that is reproducible between
operators and independent of the smoothing parameter. Both values are exported
(`Tc_Fit` and `Tc_Deriv`) so you can report the tangent value and check it against the
derivative.

---

## Input data

Quantum Design `.dat` files, found recursively under the folder you open. The reader scans
for the line containing `Temperature (K)` and treats it as the header, then keeps the
temperature and moment columns.

```
MyProject/
├── Sample_A/                 <- must match the SEM Sample_Folder name
│   └── SmFe11V_MT.dat
└── Sample_B/
    └── SmFe10V2_MT.dat
```

Heating and cooling branches are split automatically at the temperature turning point and
forced monotonic, so each branch is analysed independently.

---

## Phase windows — `config/phases.json`

Temperature search windows per phase, used by the *Auto (JSON)* button:

```json
{
  "Sm1Fe12": [500, 620],
  "Sm3Fe29": [420, 500],
  "Sm2Fe17": [370, 420],
  "FeV":     [700, 900]
}
```

For each phase the app finds the steepest drop inside the window, seeds the tangent range
there, and places the baseline range on the flat region above it. These are starting
positions — inspect and nudge them.

---

## Workflow

1. **Start New (Load Folder)** and **Load Phases (JSON)**.
2. Pick Heating or Cooling; adjust the smoothing window until d*M*/d*T* is clean without
   flattening the transition.
3. **Auto (JSON)** to seed phases, or **Add New** and set the two ranges manually.
   *Range 1* is the tangent through the transition; *Range 2* is the baseline above it.
   The orange and green bands show the active fit ranges.
4. Step through samples with the combo box or ◀ ▶. **Save Project** as you go — the project
   JSON stores the folder path, the phase dictionary and every fit range, so the session
   reopens exactly as you left it.
5. **Export Results** writes `results.csv` plus a `plots/` folder.

## Figure export

*Fig W*, *Fig H* and *Font* are physical inches and points, and the file is saved without
`bbox_inches='tight'` — so the figure keeps the exact size you asked for and the font is
true when placed in Word at 100% scale. Defaults are 6.5 × 4.5 in at 14 pt, half a
letter page with 1-inch margins. Set the same *Fig W* and *Font* in Module I and both
figures letter-match in a composite panel.

*Export M(T) panel only* drops the derivative panels, which is usually what you want for a
half-page figure. SVG and PDF keep live text (`fonttype 42`), so labels stay editable in
Illustrator.

Phase labels are rendered with subscripts automatically and a stoichiometric 1 is dropped:
`Sm1Fe12` prints as SmFe₁₂. Tc annotations are placed in the free space beside each
transition with a leader line, rather than on top of the curve.

## Merging with Module I

**Merge SEM Data** asks for the Module II `results.csv` and the Module I `.xlsx`, then
writes a workbook with four sheets:

- `0_Merge_Diagnostics` — **read this first.** Lists every sample folder or phase name
  present in one dataset but not the other. A clean run says *All Clear*.
- `1_Master_Raw` — the full row-level join.
- `2_Phase_Comp_and_Tc` — per sample and phase: mean Tc alongside mean composition.
  An outer join, so missing links stay visible instead of being silently dropped.
- `3_Sample_CrossVal` — phases seen magnetically versus phases seen spatially, side by
  side per sample, with the bulk composition attached.

Sheet 3 is where the loop closes: a phase in `MT_Detected_Phases` but absent from
`SEM_Detected_Phases` is a phase you should go back and look for in Module I's anomaly map.

The join key is exact folder-name matching (`Folder_Source` here, `Sample_Folder` there).
Phase names must also match exactly, spelling and capitalisation included.

## Notes and limits

- Overlapping transitions less than roughly 20 K apart are hard to separate by tangents;
  the diagnostics will not catch this, only your eye on the plot will.
- The nominal composition column is parsed from the filename prefix before the first
  underscore, so name files like `SmFe11V_something.dat` if you want that column populated.
