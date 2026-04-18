# Release v2026.04.18

## April 2026 refactored Krogh GUI snapshot

This release captures the current stabilized and documented state of the Krogh GUI project after the recent structural, scientific, and diagnostic improvements.

## Main highlights

### Architecture
- Incremental refactor from a large monolithic script into the reusable `krogh_app` package.
- Small startup entry point via `app.py`.
- Cleaner separation between GUI orchestration, diagnostics, reconstruction, plotting, persistence, and localization.

### Diagnostic and reconstruction workflow
- Probabilistic oxygenation diagnostic with explicit dominant alert-score drivers.
- Inverse Krogh reconstruction from blood-gas and tissue-sensor data.
- Practical uncertainty bands for mitoP50 and perfusion.
- Hidden hypoxic burden estimates below 1, 5, 10, 15, and 20 mmHg.
- Radius-conditioned interpretation across 30, 50, and 100 µm tissue assumptions.
- Clinician-oriented alert wording clarifying when stronger concern is mainly tied to swollen-tissue geometry.

### Reporting and export
- Diagnostic report export as JSON and CSV.
- English publication report export as TXT, TEX, and PDF.
- Automatic embedding of the latest Krogh reconstruction figure in publication reports.
- Improved PDF readability and report structure.

### Documentation
- Updated full project documentation.
- Updated diagnostic module documentation.
- Updated practical quickstart guide.
- Added repository-level `README.md` and `CHANGELOG.md`.

## Validation state
- Regression checks passed for reporting, persistence, reconstruction, and PDF generation at the time of release.

## Notes
This software is intended for research, educational, and decision-support style use. It is not a certified clinical device.
