"""Helpers for repeatable multi-case Krogh reconstruction benchmarks."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reconstruction import KroghReconstructor


@dataclass(frozen=True)
class ReconstructionBenchmarkCase:
    case_id: str
    label: str
    category: str
    P_inlet: float
    sensor_target: float
    P_v_target: float
    pH: float
    pCO2: float
    temp_c: float
    metabolic_target: float = 1.0
    fit_metabolic: bool = True
    include_axial: bool = True
    venous_weight: float = 0.15
    bootstrap_samples: int = 0

    def fit_kwargs(self) -> dict[str, Any]:
        return {
            "P_inlet": float(self.P_inlet),
            "sensor_target": float(self.sensor_target),
            "P_v_target": float(self.P_v_target),
            "pH": float(self.pH),
            "pCO2": float(self.pCO2),
            "temp_c": float(self.temp_c),
            "metabolic_target": float(self.metabolic_target),
            "fit_metabolic": bool(self.fit_metabolic),
            "include_axial": bool(self.include_axial),
            "venous_weight": float(self.venous_weight),
            "bootstrap_samples": int(self.bootstrap_samples),
        }


def build_default_reconstruction_benchmark_cases() -> list[ReconstructionBenchmarkCase]:
    return [
        ReconstructionBenchmarkCase(
            case_id="reference_joint_fit",
            label="Reference joint fit",
            category="normoxia",
            P_inlet=80.0,
            sensor_target=25.0,
            P_v_target=32.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
            metabolic_target=1.0,
        ),
        ReconstructionBenchmarkCase(
            case_id="normoxia_flow_reserve",
            label="Normoxic flow reserve",
            category="normoxia",
            P_inlet=96.0,
            sensor_target=33.0,
            P_v_target=41.0,
            pH=7.43,
            pCO2=37.0,
            temp_c=36.9,
            metabolic_target=0.95,
        ),
        ReconstructionBenchmarkCase(
            case_id="low_perfusion_stress",
            label="Low-perfusion stress case",
            category="stress",
            P_inlet=72.0,
            sensor_target=19.0,
            P_v_target=27.0,
            pH=7.36,
            pCO2=43.0,
            temp_c=37.0,
            metabolic_target=1.1,
        ),
        ReconstructionBenchmarkCase(
            case_id="higher_metabolic_demand",
            label="Higher metabolic demand",
            category="stress",
            P_inlet=86.0,
            sensor_target=28.0,
            P_v_target=35.0,
            pH=7.4,
            pCO2=39.0,
            temp_c=37.0,
            metabolic_target=1.25,
        ),
        ReconstructionBenchmarkCase(
            case_id="borderline_hypercapnic_edge",
            label="Borderline hypercapnic edge",
            category="edge",
            P_inlet=68.0,
            sensor_target=17.0,
            P_v_target=24.0,
            pH=7.31,
            pCO2=51.0,
            temp_c=37.2,
            metabolic_target=1.08,
        ),
        ReconstructionBenchmarkCase(
            case_id="acidotic_low_sensor_edge",
            label="Acidotic low-sensor edge",
            category="edge",
            P_inlet=62.0,
            sensor_target=13.0,
            P_v_target=20.0,
            pH=7.27,
            pCO2=56.0,
            temp_c=37.0,
            metabolic_target=1.15,
        ),
        ReconstructionBenchmarkCase(
            case_id="combined_severe_transport_stress",
            label="Combined severe transport stress",
            category="stress",
            P_inlet=54.0,
            sensor_target=8.5,
            P_v_target=14.0,
            pH=7.24,
            pCO2=58.0,
            temp_c=37.0,
            metabolic_target=1.22,
            venous_weight=0.20,
        ),
        ReconstructionBenchmarkCase(
            case_id="alkalotic_reserve_shift",
            label="Alkalotic reserve shift",
            category="edge",
            P_inlet=100.0,
            sensor_target=35.0,
            P_v_target=43.0,
            pH=7.48,
            pCO2=33.0,
            temp_c=36.8,
            metabolic_target=0.92,
        ),
    ]


class ReconstructionBenchmarkRunner:
    """Runs repeatable wall-clock benchmarks for Krogh reconstructions."""

    def __init__(
        self,
        cases: list[ReconstructionBenchmarkCase] | None = None,
        reconstructor: KroghReconstructor | Any | None = None,
    ) -> None:
        if reconstructor is None:
            import krogh_GUI as runtime

            reconstructor = KroghReconstructor(
                solve_axial_capillary_po2=runtime.solve_axial_capillary_po2,
                effective_p50=runtime.effective_p50,
                radial_weights=runtime.radial_weights,
                r_vec=runtime.r_vec,
                z_eval=runtime.z_eval,
                R_cap=runtime.R_cap,
                R_tis=runtime.R_tis,
                L_cap=runtime.L_cap,
                get_numeric_settings=runtime.get_numeric_settings,
                temporary_numeric_settings=runtime.temporary_numeric_settings,
                build_fast_numeric_settings=runtime.build_fast_numeric_settings,
            )
        self.reconstructor = reconstructor
        self.cases = list(cases) if cases is not None else build_default_reconstruction_benchmark_cases()

    def run_case(self, case: ReconstructionBenchmarkCase, *, search_strategy: str = "optimized") -> dict[str, Any]:
        start = time.perf_counter()
        fit = self.reconstructor.fit_joint_parameters(
            **case.fit_kwargs(),
            search_strategy=search_strategy,
        )
        elapsed_s = float(time.perf_counter() - start)
        uncertainty = dict(fit.get("uncertainty", {}))
        return {
            "case_id": case.case_id,
            "label": case.label,
            "category": case.category,
            "search_strategy": str(search_strategy),
            "elapsed_s": elapsed_s,
            "objective": float(fit["objective"]),
            "perfusion_factor": float(fit["perfusion_factor"]),
            "P_half_fit": float(fit["P_half_fit"]),
            "metabolic_rate_rel": float(fit.get("metabolic_rate_rel", 1.0)),
            "sensor_error": float(fit.get("sensor_error", np.nan)),
            "venous_error": float(fit.get("venous_error", np.nan)),
            "fit_warning": bool(fit.get("fit_warning", False)),
            "fit_boundary_hit": bool(fit.get("fit_boundary_hit", False)),
            "candidate_count": int(uncertainty.get("candidate_count", 0)),
            "identifiability": str(uncertainty.get("identifiability", "unknown")),
        }

    def build_comparison_rows(self, optimized_rows: list[dict[str, Any]], legacy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        legacy_by_case = {row["case_id"]: row for row in legacy_rows}
        comparison_rows: list[dict[str, Any]] = []
        for optimized in optimized_rows:
            legacy = legacy_by_case.get(optimized["case_id"])
            if legacy is None:
                continue
            optimized_elapsed = float(optimized["elapsed_s"])
            legacy_elapsed = float(legacy["elapsed_s"])
            comparison_rows.append(
                {
                    "case_id": str(optimized["case_id"]),
                    "label": str(optimized["label"]),
                    "category": str(optimized.get("category", "unknown")),
                    "optimized_elapsed_s": optimized_elapsed,
                    "legacy_elapsed_s": legacy_elapsed,
                    "elapsed_saved_s": float(legacy_elapsed - optimized_elapsed),
                    "speedup_ratio": float(legacy_elapsed / max(optimized_elapsed, 1e-9)),
                    "optimized_objective": float(optimized["objective"]),
                    "legacy_objective": float(legacy["objective"]),
                    "objective_delta": float(optimized["objective"] - legacy["objective"]),
                    "optimized_identifiability": str(optimized["identifiability"]),
                    "legacy_identifiability": str(legacy["identifiability"]),
                }
            )
        return comparison_rows

    def _build_comparison_summary(self, comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not comparison_rows:
            return {
                "case_count": 0,
                "mean_speedup_ratio": np.nan,
                "median_speedup_ratio": np.nan,
                "mean_elapsed_saved_s": np.nan,
            }

        speedups = np.array([row["speedup_ratio"] for row in comparison_rows], dtype=float)
        savings = np.array([row["elapsed_saved_s"] for row in comparison_rows], dtype=float)
        return {
            "case_count": int(len(comparison_rows)),
            "mean_speedup_ratio": float(np.mean(speedups)),
            "median_speedup_ratio": float(np.median(speedups)),
            "mean_elapsed_saved_s": float(np.mean(savings)),
        }

    def run(self, *, verbose: bool = False) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        legacy_rows: list[dict[str, Any]] = []
        for index, case in enumerate(self.cases, start=1):
            if verbose:
                print(f"[benchmark] case {index}/{len(self.cases)}: {case.case_id}")
            row = self.run_case(case, search_strategy="optimized")
            rows.append(row)
            if verbose:
                print(
                    f"[benchmark]   -> {row['elapsed_s']:.3f} s | objective {row['objective']:.4f} | "
                    f"perf {row['perfusion_factor']:.3f}x | mitoP50 {row['P_half_fit']:.3f}"
                )
                print("[benchmark]   -> running legacy_grid comparison")
            legacy_row = self.run_case(case, search_strategy="legacy_grid")
            legacy_rows.append(legacy_row)
            if verbose:
                print(
                    f"[benchmark]   -> legacy {legacy_row['elapsed_s']:.3f} s | "
                    f"speedup {legacy_row['elapsed_s'] / max(row['elapsed_s'], 1e-9):.2f}x"
                )

        elapsed_values = np.array([row["elapsed_s"] for row in rows], dtype=float)
        comparison_rows = self.build_comparison_rows(rows, legacy_rows)
        summary = {
            "case_count": int(len(rows)),
            "mean_elapsed_s": float(np.mean(elapsed_values)) if len(rows) else np.nan,
            "median_elapsed_s": float(np.median(elapsed_values)) if len(rows) else np.nan,
            "max_elapsed_s": float(np.max(elapsed_values)) if len(rows) else np.nan,
            "min_elapsed_s": float(np.min(elapsed_values)) if len(rows) else np.nan,
            "warning_count": int(sum(1 for row in rows if row["fit_warning"])),
            "boundary_hit_count": int(sum(1 for row in rows if row["fit_boundary_hit"])),
        }
        return {
            "summary": summary,
            "cases": rows,
            "legacy_cases": legacy_rows,
            "comparison_summary": self._build_comparison_summary(comparison_rows),
            "comparison": comparison_rows,
        }

    def save_csv_report(self, results: dict[str, Any], path: str | Path) -> None:
        rows = list(results.get("cases", []))
        if not rows:
            raise ValueError("No benchmark rows available to write.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def save_json_report(self, results: dict[str, Any], path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def format_summary_text(self, results: dict[str, Any]) -> str:
        summary = dict(results.get("summary", {}))
        comparison_summary = dict(results.get("comparison_summary", {}))
        return (
            f"Reconstruction benchmark summary: {summary.get('case_count', 0)} cases | "
            f"mean {summary.get('mean_elapsed_s', float('nan')):.3f} s | "
            f"median {summary.get('median_elapsed_s', float('nan')):.3f} s | "
            f"max {summary.get('max_elapsed_s', float('nan')):.3f} s | "
            f"warnings {summary.get('warning_count', 0)} | boundary hits {summary.get('boundary_hit_count', 0)} | "
            f"mean legacy speedup {comparison_summary.get('mean_speedup_ratio', float('nan')):.2f}x."
        )


def run_and_save_default_reconstruction_benchmark(output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    runner = ReconstructionBenchmarkRunner()
    results = runner.run(verbose=True)
    runner.save_csv_report(results, destination / "reconstruction_benchmark.csv")
    runner.save_json_report(results, destination / "reconstruction_benchmark.json")
    runner.save_csv_report({"cases": results["comparison"]}, destination / "reconstruction_benchmark_comparison.csv")
    runner.save_json_report(
        {
            "summary": results.get("comparison_summary", {}),
            "cases": results.get("comparison", []),
        },
        destination / "reconstruction_benchmark_comparison.json",
    )
    return results


__all__ = [
    "ReconstructionBenchmarkCase",
    "ReconstructionBenchmarkRunner",
    "build_default_reconstruction_benchmark_cases",
    "run_and_save_default_reconstruction_benchmark",
]