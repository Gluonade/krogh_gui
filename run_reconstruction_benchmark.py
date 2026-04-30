"""Convenience entry point for the multi-case Krogh reconstruction benchmark."""

from __future__ import annotations

from pathlib import Path

from krogh_app.benchmarking import ReconstructionBenchmarkRunner, run_and_save_default_reconstruction_benchmark


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "Diagnostic reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting reconstruction benchmark across representative cases.")
    print("Progress lines should appear case by case; the full run can take a few minutes.")

    results = run_and_save_default_reconstruction_benchmark(output_dir)
    print(ReconstructionBenchmarkRunner().format_summary_text(results))
    print(f"Saved reports to: {output_dir}")
    print(
        "Generated: reconstruction_benchmark.csv, reconstruction_benchmark.json, "
        "reconstruction_benchmark_comparison.csv, and reconstruction_benchmark_comparison.json"
    )


if __name__ == "__main__":
    main()