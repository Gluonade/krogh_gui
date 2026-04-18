# GitHub Release Text for v2026.04.18

## Title
v2026.04.18 — Refactored Krogh GUI snapshot with improved diagnostic interpretation

## Release body
This release captures the current stabilized April 2026 state of the Krogh GUI project after substantial work on structure, diagnostics, reconstruction, reporting, and documentation.

### Highlights
- Refactored the project into a cleaner modular structure centered around the `krogh_app` package.
- Preserved the validated scientific workflow while making the codebase easier to maintain and extend.
- Expanded the diagnostic workflow with dominant alert-driver explanations.
- Added inverse Krogh reconstruction with practical uncertainty reporting.
- Added hidden hypoxic burden summaries below clinically relevant PO2 thresholds.
- Added radius-aware interpretation for 30, 50, and 100 µm tissue-cylinder assumptions.
- Improved publication-style reporting with English TXT, TEX, and PDF export.
- Added embedded reconstruction figures to publication reports.
- Updated the documentation and quickstart material to reflect the new architecture and scientific interpretation layer.

### Scientific interpretation improvements
A central improvement of this release is that elevated alerts are now interpreted more transparently. The software no longer presents only a color category. It now clarifies:
- which features drove the alert,
- which hidden assumptions were required to reproduce the measured pattern,
- how much tissue is estimated to lie below critical hypoxic thresholds,
- and whether stronger concern mainly appears under a larger or swollen-tissue radius assumption.

### Validation
The project was rechecked with regression and compile validation during this release cycle, including reporting, persistence, reconstruction, and PDF export workflows.

### Notes
This software is intended for research, educational, and decision-support style use. It is not a certified clinical device.
