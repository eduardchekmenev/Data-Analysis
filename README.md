# NMR Data Analysis Pipeline

A collection of Python GUI tools for processing and analyzing Nuclear Magnetic Resonance (NMR) spectroscopy data — primarily for time-series metabolomics experiments in cell biology research (cancer and yeast models).

## Overview

Raw NMR data flows through a numbered pipeline of PyQt5-based desktop tools, each handling a discrete processing step from raw spectra to statistical output.

```
Raw NMR Data (Bruker / Spinsolve / Prodigy Dynamics)
       │
       ▼
[Raw Data Visualization]  ←── Load, inspect, and integrate/deconvolve raw spectra
       │
       ▼
[0] Duration Finder       ←── Extract timing parameters from instrument config files
       │
       ▼
[1] SVD FID Analysis      ←── Denoise Free Induction Decay data via SVD
       │
       ▼
[2] SVD Output Processing ←── Post-process denoised output, extract metadata
       │
       ▼
[3] Time-Series Fitting   ←── Fit metabolite peaks to kinetic models (cancer / yeast)
       │
       ▼
[4] Cell Stats Analysis   ←── Statistics, PCA, heatmaps, multi-test correction
       │
       ▼
      Results / Figures
```

## Directory Structure

```
Data-Analysis/
├── Working Data Pipeline/     # [LOCKED] Core numbered pipeline scripts (steps 0–6)
├── Raw Data Visualization/    # [LOCKED] Instrument-specific data loading GUIs
├── Archive/                   # Superseded versions organized by function
├── developmental/             # Free development area — prototype new tools here
└── contrib/                   # User contributions, adaptations, and extensions
```

`Working Data Pipeline/` and `Raw Data Visualization/` are locked — changes go through the core maintainers. Use `developmental/` and `contrib/` for open development (see their READMEs for guidelines).

## Scripts

### Raw Data Visualization

Instrument-specific GUIs for loading and inspecting raw NMR spectra before pipeline processing.

| Script | Instrument | Notes |
|--------|-----------|-------|
| `PD_DataAnalysis-gui4.py` | Prodigy Dynamics | Multi-folder picker; basic integration or multi-peak deconvolution |
| `Spinsolve_DataAnalysis_gui-4.py` | Spinsolve | Exponential apodization support |
| `Bruker_DataAnalysis_gui.py` | Bruker | Standard Bruker data format |

### Working Data Pipeline

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `0-Duration_Finder-1.py` | Parse `acqu.par` config files to extract experiment timing/duration parameters |
| 1 | `1-SVD_FIDanalysis_1-5-gui2.py` | SVD-based denoising of FID data across multiple folders; custom paste/delete table UI |
| 2 | `2-SVDout_proc_gui-3.py` | Post-process SVD output; extract dates from filenames; further data refinement |
| 3 | `3-TimeSeriesFit_gui-5-cancer.py` | Fit time-series metabolite data to kinetic models — cancer cell variant |
| 3 | `3-TimeSeriesFit_gui-6-yeast.py` | Fit time-series metabolite data to kinetic models — yeast variant |
| 4 | `4-Data_CellStats-7.py` | Cell-level statistics: t-tests, Bonferroni correction, PCA, heatmaps |
| 5 | `5-Denoise_Visual-1.py` | Visualize denoising results |
| 6 | `6-MM_Fitting-2.py` | Michaelis-Menten or molecule-level fitting |

## Tech Stack

- **Language:** Python 3
- **GUI:** PyQt5
- **Scientific computing:** NumPy, SciPy, Pandas
- **NMR processing:** nmrglue
- **Visualization:** Matplotlib, Seaborn (embedded in Qt via FigureCanvas)
- **Machine learning:** scikit-learn (PCA, StandardScaler)
- **Statistics:** scipy.stats, statsmodels (Bonferroni multiple-test correction)
- **Optimization:** scipy.optimize (curve_fit, least_squares)

## Archive

The `Archive/` folder contains earlier iterations of scripts, organized by function:

- `Cell Stats Analysis/` — older stats tools
- `Data Manipulation/` — earlier data-wrangling scripts
- `Metabolite Fitting/` — previous fitting approaches
- `SVD Output Processing/` — earlier post-processing scripts
- `SVD_Denoising/` — earlier denoising implementations
- `Raw Data Visualization/` — older visualization GUIs

## Contributing

| Folder | Who it's for | Rules |
|--------|-------------|-------|
| `developmental/` | Anyone — prototype new tools, test modifications | No restrictions; organize in a named subfolder |
| `contrib/` | Anyone — share scripts that extend or wrap the pipeline | Keep contributions self-contained in a named subfolder |
| `Working Data Pipeline/` | Core maintainers only | Locked |
| `Raw Data Visualization/` | Core maintainers only | Locked |

See `developmental/README.md` and `contrib/README.md` for detailed guidelines.
