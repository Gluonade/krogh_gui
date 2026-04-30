# Krogh GUI

A Tkinter-based research and diagnostic application for Krogh-cylinder oxygen transport analysis, parameter sweeps, probabilistic oxygenation assessment, and inverse reconstruction from blood-gas and tissue-sensor data.

## Current project state

As of 18 April 2026, the project has been substantially improved in three major ways:

1. **Architecture and maintainability**
   - The application is no longer a single monolithic script.
   - Core logic has been reorganized into the `krogh_app` package.
   - `app.py` now serves as the small startup entry point.
   - `krogh_GUI.py` remains as the compatibility-oriented main controller.

2. **Scientific transparency and model interpretation**
   - The diagnostic section now reports dominant alert-score drivers.
   - Krogh reconstruction now provides practical uncertainty bands for fitted mitoP50 and perfusion.
   - Hidden hypoxic burden is quantified below 1, 5, 10, 15, and 20 mmHg.
   - Radius-sensitive interpretation is now available for 30, 50, and 100 µm tissue-cylinder assumptions.

3. **Reporting and documentation**
   - Diagnostic reports can be exported as JSON or CSV.
   - English publication reports can be exported as TXT, TEX, or PDF.
   - The latest reconstruction figure is embedded automatically in publication reports when available.
   - Documentation files were updated to reflect the recent April 2026 improvements.

## Main capabilities

- Single-case Krogh-cylinder simulation
- One-dimensional and two-dimensional parameter sweeps
- Adjustable numerical solver settings
- Probabilistic oxygenation diagnostic evaluation
- Inverse reconstruction from diagnostic inputs
- Radius-conditioned interpretation of alert severity
- Case save/load and report export
- English publication-ready report generation

## Project structure

```text
Tkinter_GUI/
├── app.py
├── krogh_GUI.py
├── oxygenation_diagnostic_mvp.py
├── krogh_app/
│   ├── constants.py
│   ├── diagnostics.py
│   ├── localization.py
│   ├── persistence.py
│   ├── plotting.py
│   ├── reconstruction.py
│   ├── series.py
│   ├── types.py
│   └── ui/
├── tests/
├── krogh_GUI_documentation.tex
├── krogh_GUI_quickstart.tex
└── oxygenation_diagnostic_mvp_documentation.tex
```

## How to run

Activate the virtual environment and start the application:

```bash
source .venv/bin/activate
python app.py
```

## Synthetic validation

A reusable synthetic validation runner is now included for controlled forward-plus-inverse checks using fictive but physiologically plausible cases. It generates case-wise CSV and JSON summaries in the `Diagnostic reports/` folder and is intended as the next practical validation step before harder real-world benchmarking.

Run it with:

```bash
source .venv/bin/activate
python run_synthetic_validation.py
```

A short user guide is also available in the project root:

- [synthetic_validation_guide.pdf](synthetic_validation_guide.pdf)

It explains expected runtime behaviour, when to wait versus intervene, where the result files are written, and how to interpret the validation outputs.

## Reconstruction benchmark

A reusable multi-case reconstruction benchmark is also available. It times several representative joint-fit scenarios and writes CSV and JSON summaries into the `Diagnostic reports/` folder.

Run it with:

```bash
source .venv/bin/activate
python run_reconstruction_benchmark.py
```

The benchmark is intended for practical before/after performance comparisons when the reconstruction search strategy or solver numerics change.

## Documentation

The main documentation files are:

- `krogh_GUI_documentation.pdf` — full mathematical, numerical, and implementation documentation
- `oxygenation_diagnostic_mvp_documentation.pdf` — diagnostic and fitting logic documentation
- `krogh_GUI_quickstart.pdf` — short practical quickstart guide
- `CHANGELOG.md` — concise summary of the recent project milestones

## Recent highlights

Recent verified changes include:

- modular refactor into reusable service modules
- persistence of diagnostic and reconstruction state
- structured English publication reports
- automatic PDF generation with embedded figures
- clinician-oriented interpretation text
- explicit radius-conditioned alert explanation
- corrected default diagnostic sensor value of 25 mmHg

## Recommended next scientific priorities

- formalize uncertainty intervals with bootstrap or profile-likelihood analysis
- add practical identifiability checks for fitted parameters
- evaluate a reduced Bayesian inverse mode for posterior mapping in the low-dimensional fit space
- defer pulsatile forcing and explicit capillary-network models until time-resolved or morphometric data justify the extra complexity

## Intended use

This software is designed for research, mechanistic exploration, and decision-support style interpretation. It is not a certified clinical device and should be interpreted together with physiological expertise and the explicit assumptions stated in the reports.
