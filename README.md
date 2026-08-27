# When Does the Graph Actually Help? - Code and Data

Anonymous code and data release accompanying the LoG 2026 extended
abstract submission "When Does the Graph Actually Help? Decomposing
Feature and Structure Signal in Deep Graph Clustering."

All 2,028 experiment runs reported in the paper are included as CSVs,
so every number and figure can be regenerated without re-running
anything. Full re-runs are also supported (about one hour on a laptop
CPU for the synthetic sweep).

## Contents

Scripts are named `dayN_*` in build order; each later script imports
helpers from earlier ones.

| File | What it is |
|---|---|
| `generator.py` | Attributed planted-partition SBM with independent structure and feature knobs (the paper's instrument) |
| `day1_pilot.py` | Knob validation + method wrappers (Louvain, Leiden, k-means) |
| `day2_minigrid.py` | 3x3 pilot grid (cheap-method winner map) |
| `day3_dmon.py` | DMoN, fixed-budget configuration (GCN + skip encoder) |
| `day4_runner.py` | Full 8x8 x 5-seed sweep, all methods; resumable; writes `results.csv`. Also defines MinCut |
| `day5_analysis.py` | Heatmaps, winner map, delta maps from `results.csv` |
| `day6_realdata.py` | Cora / Citeseer / Amazon Photo + channel ablations (noise-X, rewired); writes `realdata.csv` |
| `day7_fairness.py` | Reference-faithful DMoN (official objective, dropout, epochs); fairness comparison; writes `fairness.csv` |
| `day8_refresh.py` | Reference DMoN across the full sweep + reference-config real-data ablations |
| `paper_figures.py` | Publication figures (Fig. 1 and Fig. 2) |
| `results.csv` | 1,920 synthetic runs (8x8 grid, 5 seeds, 6 methods) |
| `realdata.csv` | 81 real-data runs (3 datasets, 3 seeds, methods + ablations) |
| `fairness.csv` | 27 reference-config validation runs |

## Author-response artifacts (added during the review discussion)

| File | What it is |
|---|---|
| `REBUTTAL.md` | Full point-by-point author responses to the reviews |
| `rebuttal_lib.py` | Multi-layer reference DMoN (depth=1 reproduces `dmon_ref` exactly; checked by `rebuttal_pilot.py`) |
| `rebuttal_pilot.py` | Fidelity check and per-config timings |
| `rebuttal_runner.py` | Depth 2/3 sweeps (DMoN and MinCut), hyperparameter grid, real-data depth; writes `rebuttal57.csv` |
| `rebuttal_newdata.py` | Amazon Computers + Coauthor CS, full Table-1 battery; writes `rebuttal_newdata.csv` |
| `rebuttal_analysis.py` | Recomputes the headline numbers quoted in `REBUTTAL.md` from the CSVs |
| `rebuttal57.csv` | 3,538 rebuttal runs (depth, tuning, real-data depth) |
| `rebuttal_newdata.csv` | 42 rebuttal runs (two additional datasets) |

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Regenerate the paper's figures from the shipped CSVs

```
python paper_figures.py          # writes fig1_map.* and fig2_deltas.*
python day5_analysis.py          # full heatmap panel + delta maps
```

## Full reproduction from scratch

```
python day4_runner.py            # synthetic sweep (~1 h laptop CPU)
python day8_refresh.py           # reference-config DMoN column + real ablations
python day6_realdata.py          # real-data table (downloads datasets)
python day7_fairness.py          # reference-config validation
python paper_figures.py
```

Every runner is resumable: rerun the same command and completed
(setting, seed, method) rows are skipped.

## Notes

- The reference DMoN configuration follows the official implementation
  (loss = spectral + collapse with no orthogonality term, dropout 0.5,
  shared-kernel skip, 1,000 epochs, Adam 1e-3) and reproduces the
  published Cora score within seed noise (per-seed NMI 0.489, 0.445,
  0.444 vs published 0.488; mean 0.459).
- In `realdata.csv`, methods prefixed `dmonref_` are the channel
  ablations run at the reference configuration (`dmon_ref`); the
  `dmon_`-prefixed ablations use the fixed-budget configuration.
- Real-data labels follow the standard convention of using class
  labels as community proxies.
- Datasets download automatically via PyTorch Geometric on first run.
