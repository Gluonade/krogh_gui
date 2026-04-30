"""Convenience entry point for the synthetic validation suite."""

from __future__ import annotations

from pathlib import Path

from krogh_app.validation import SyntheticValidationRunner, run_and_save_default_validation


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "Diagnostic reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting synthetic validation. Progress lines should appear case by case.")
    print("If nothing new appears for a very long time and CPU activity is clearly idle, then interrupt and inspect.")

    results = run_and_save_default_validation(str(output_dir))
    print(SyntheticValidationRunner().format_summary_text(results))
    print(f"Saved reports to: {output_dir}")
    print("Also generated: trend-check and robustness validation reports.")


if __name__ == "__main__":
    main()
