# Changelog

## 18 April 2026

### Documentation and communication
- Updated the main project documentation to describe the latest architectural refactor and the diagnostic workflow extensions.
- Rebuilt the LaTeX documentation PDFs after a consistency and style pass.
- Added a top-level project overview in `README.md`.

### Diagnostic workflow improvements
- Expanded the diagnostic interpretation to report dominant alert-score drivers.
- Added clinician-oriented wording to explain how the alert should be interpreted.
- Corrected the default tissue sensor example value to 25 mmHg.

### Krogh reconstruction improvements
- Added practical uncertainty bands for fitted mitoP50 and perfusion.
- Added best-fit assumption summaries for inverse reconstruction.
- Added hidden hypoxic burden estimates below 1, 5, 10, 15, and 20 mmHg.
- Added radius-sensitive comparison across 30, 50, and 100 µm tissue-cylinder assumptions.
- Added explicit notes clarifying when a stronger alert is mainly supported under a larger or swollen-tissue geometry.

### Reporting and export
- Added structured diagnostic report export in JSON and CSV format.
- Added English publication report export in TXT, TEX, and PDF format.
- Added automatic embedding of the latest reconstruction figure into publication reports.
- Improved PDF layout to reduce overlap and improve readability.

### Project architecture
- Continued the modular refactor by keeping the entry point small in `app.py` while moving reusable logic into the `krogh_app` package.
- Preserved legacy GUI continuity while improving maintainability and auditability.

---

## 16 April 2026

### Refactor milestones
- Established the incremental package-based refactor structure.
- Extracted constants, typing, diagnostics, reconstruction, plotting, series, persistence, localization, and focused UI helpers into modular files.
- Preserved validated scientific behavior while reducing monolithic complexity.

### Stability and verification
- Repeated regression and compile checks were used throughout the refactor to keep scientific behavior stable.
