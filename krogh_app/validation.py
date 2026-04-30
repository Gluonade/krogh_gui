"""Synthetic validation helpers for the Krogh GUI project.

This module supports the next practical validation phase of the project:
use physiologically plausible synthetic cases to stress-test the forward model,
the diagnostic interpretation, and the inverse reconstruction under controlled
conditions before real-world reference datasets are available.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .diagnostics import DiagnosticEngine
from .reconstruction import KroghReconstructor
from .types import DiagnosticRunInput


@dataclass(frozen=True)
class SyntheticValidationCase:
    case_id: str
    label: str
    category: str
    P_inlet: float
    P_half: float
    pH: float
    pCO2: float
    temp_c: float
    perfusion_factor: float
    include_axial_diffusion: bool = False
    hemoglobin_g_dl: float = 13.5
    sensor_tolerance: float = 3.0
    venous_tolerance: float = 4.0
    p_half_relative_tolerance: float = 0.60
    perfusion_relative_tolerance: float = 0.35

    def forward_params(self) -> dict[str, Any]:
        return {
            "P_inlet": float(self.P_inlet),
            "P_half": float(self.P_half),
            "pH": float(self.pH),
            "pCO2": float(self.pCO2),
            "temp_c": float(self.temp_c),
            "perfusion_factor": float(self.perfusion_factor),
            "include_axial_diffusion": bool(self.include_axial_diffusion),
        }


def _reconstruction_grids() -> tuple[np.ndarray, np.ndarray]:
    perf_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.35, 5.0, 15))))
    p_half_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.05, 300.0, 22))))
    return perf_candidates, p_half_candidates


def build_default_synthetic_cases() -> list[SyntheticValidationCase]:
    perf_grid, p_half_grid = _reconstruction_grids()

    return [
        SyntheticValidationCase(
            case_id="normoxia_reference",
            label="Reference normoxia",
            category="normoxia",
            P_inlet=95.0,
            P_half=1.0,
            pH=7.40,
            pCO2=40.0,
            temp_c=37.0,
            perfusion_factor=1.0,
        ),
        SyntheticValidationCase(
            case_id="high_flow_reserve",
            label="High-flow reserve",
            category="normoxia",
            P_inlet=95.0,
            P_half=1.0,
            pH=7.42,
            pCO2=38.0,
            temp_c=37.0,
            perfusion_factor=float(perf_grid[-3]),
        ),
        SyntheticValidationCase(
            case_id="low_perfusion_hidden_hypoxia",
            label="Low perfusion with hidden burden",
            category="low_perfusion",
            P_inlet=78.0,
            P_half=1.0,
            pH=7.36,
            pCO2=43.0,
            temp_c=37.0,
            perfusion_factor=float(perf_grid[2]),
        ),
        SyntheticValidationCase(
            case_id="mito_shift_mild_hypoxia",
            label="Higher mitoP50 stress",
            category="mild_hypoxia",
            P_inlet=82.0,
            P_half=float(p_half_grid[11]),
            pH=7.38,
            pCO2=41.0,
            temp_c=37.0,
            perfusion_factor=1.0,
        ),
        SyntheticValidationCase(
            case_id="acid_hypercapnic_strain",
            label="Acidotic and hypercapnic strain",
            category="mild_hypoxia",
            P_inlet=70.0,
            P_half=float(p_half_grid[9]),
            pH=7.28,
            pCO2=55.0,
            temp_c=37.0,
            perfusion_factor=float(perf_grid[4]),
        ),
        SyntheticValidationCase(
            case_id="combined_severe_stress",
            label="Combined severe transport stress",
            category="severe_hypoxia",
            P_inlet=52.0,
            P_half=float(p_half_grid[12]),
            pH=7.24,
            pCO2=58.0,
            temp_c=37.0,
            perfusion_factor=float(perf_grid[1]),
            sensor_tolerance=4.0,
            venous_tolerance=5.0,
            p_half_relative_tolerance=0.80,
            perfusion_relative_tolerance=0.45,
        ),
    ]


def build_extended_synthetic_cases() -> list[SyntheticValidationCase]:
    perf_grid, p_half_grid = _reconstruction_grids()
    return build_default_synthetic_cases() + [
        SyntheticValidationCase(
            case_id="borderline_anemia_case",
            label="Borderline anemia stress",
            category="anemia_stress",
            P_inlet=74.0,
            P_half=1.0,
            pH=7.36,
            pCO2=44.0,
            temp_c=37.0,
            perfusion_factor=1.0,
            hemoglobin_g_dl=9.8,
            p_half_relative_tolerance=0.75,
        ),
        SyntheticValidationCase(
            case_id="temperature_shift_case",
            label="Thermal stress borderline case",
            category="temperature_stress",
            P_inlet=72.0,
            P_half=float(p_half_grid[9]),
            pH=7.34,
            pCO2=46.0,
            temp_c=39.2,
            perfusion_factor=float(perf_grid[5]),
            p_half_relative_tolerance=0.75,
        ),
        SyntheticValidationCase(
            case_id="alkalotic_reserve_case",
            label="Alkalotic reserve case",
            category="reserve_shift",
            P_inlet=100.0,
            P_half=1.0,
            pH=7.48,
            pCO2=33.0,
            temp_c=36.8,
            perfusion_factor=float(perf_grid[8]),
        ),
    ]


def _load_runtime_hooks() -> dict[str, Any]:
    import krogh_GUI as runtime

    return {
        "run_single_case": runtime.run_single_case,
        "solve_axial_capillary_po2": runtime.solve_axial_capillary_po2,
        "effective_p50": runtime.effective_p50,
        "get_numeric_settings": runtime.get_numeric_settings,
        "temporary_numeric_settings": runtime.temporary_numeric_settings,
        "build_fast_numeric_settings": runtime.build_fast_numeric_settings,
        "radial_weights": runtime.radial_weights,
        "r_vec": runtime.r_vec,
        "z_eval": runtime.z_eval,
        "R_cap": runtime.R_cap,
        "R_tis": runtime.R_tis,
        "L_cap": runtime.L_cap,
    }


class SyntheticValidationRunner:
    """Runs synthetic forward-plus-inverse consistency checks."""

    def __init__(self, cases: list[SyntheticValidationCase] | None = None) -> None:
        hooks = _load_runtime_hooks()
        self.run_single_case = hooks["run_single_case"]
        self.diagnostic_engine = DiagnosticEngine()
        self.reconstructor = KroghReconstructor(
            solve_axial_capillary_po2=hooks["solve_axial_capillary_po2"],
            effective_p50=hooks["effective_p50"],
            radial_weights=hooks["radial_weights"],
            r_vec=hooks["r_vec"],
            z_eval=hooks["z_eval"],
            R_cap=hooks["R_cap"],
            R_tis=hooks["R_tis"],
            L_cap=hooks["L_cap"],
            get_numeric_settings=hooks["get_numeric_settings"],
            temporary_numeric_settings=hooks["temporary_numeric_settings"],
            build_fast_numeric_settings=hooks["build_fast_numeric_settings"],
        )
        self.cases = list(cases) if cases is not None else build_default_synthetic_cases()

        if not self.diagnostic_engine.is_available:
            raise RuntimeError("DiagnosticEngine is not available for synthetic validation.")

    def _rng_for_case(self, case: SyntheticValidationCase, sensor_noise_std: float, venous_sat_noise_std: float):
        seed = (
            sum(ord(ch) for ch in case.case_id)
            + int(round(100.0 * float(sensor_noise_std)))
            + 97 * int(round(100.0 * float(venous_sat_noise_std)))
        ) % (2**32 - 1)
        return np.random.default_rng(seed)

    def _simulate_observed_measurements(
        self,
        case: SyntheticValidationCase,
        *,
        sensor_noise_std: float = 0.0,
        venous_sat_noise_std: float = 0.0,
    ) -> dict[str, Any]:
        truth = self.run_single_case(**case.forward_params())
        rng = self._rng_for_case(case, sensor_noise_std, venous_sat_noise_std)

        sensor_observed = max(0.0, float(truth["PO2_sensor_avg"]) + float(rng.normal(0.0, float(sensor_noise_std))))
        venous_sat_observed = float(np.clip(
            float(truth["S_v_percent"]) + float(rng.normal(0.0, float(venous_sat_noise_std))),
            1.0,
            99.0,
        ))

        return {
            "truth": truth,
            "sensor_observed": sensor_observed,
            "venous_sat_observed": venous_sat_observed,
        }

    def evaluate_case(
        self,
        case: SyntheticValidationCase,
        *,
        sensor_noise_std: float = 0.0,
        venous_sat_noise_std: float = 0.0,
        bootstrap_samples: int = 8,
    ) -> dict[str, Any]:
        observed = self._simulate_observed_measurements(
            case,
            sensor_noise_std=sensor_noise_std,
            venous_sat_noise_std=venous_sat_noise_std,
        )
        truth = observed["truth"]

        diagnostic_input = DiagnosticRunInput(
            po2=float(case.P_inlet),
            pco2=float(case.pCO2),
            pH=float(case.pH),
            temperature_c=float(case.temp_c),
            sensor_po2=float(observed["sensor_observed"]),
            hemoglobin_g_dl=float(case.hemoglobin_g_dl),
            venous_sat_percent=float(observed["venous_sat_observed"]),
        )
        diagnostic_result = self.diagnostic_engine.evaluate(diagnostic_input)
        P_v_target = self.diagnostic_engine.venous_target_po2(
            pH=float(case.pH),
            pco2=float(case.pCO2),
            temperature_c=float(case.temp_c),
            venous_sat_percent=float(observed["venous_sat_observed"]),
        )

        fit = self.reconstructor.fit_joint_parameters(
            P_inlet=float(case.P_inlet),
            sensor_target=float(observed["sensor_observed"]),
            P_v_target=float(P_v_target),
            pH=float(case.pH),
            pCO2=float(case.pCO2),
            temp_c=float(case.temp_c),
            include_axial=bool(case.include_axial_diffusion),
            venous_weight=0.55,
            bootstrap_samples=int(bootstrap_samples),
        )

        p_half_rel_error = abs(float(fit["P_half_fit"]) - float(case.P_half)) / max(abs(float(case.P_half)), 1e-6)
        perfusion_rel_error = abs(float(fit["perfusion_factor"]) - float(case.perfusion_factor)) / max(abs(float(case.perfusion_factor)), 1e-6)
        reconstruction_success = bool(
            float(fit["sensor_error"]) <= float(case.sensor_tolerance)
            and float(fit["venous_error"]) <= float(case.venous_tolerance)
        )
        parameter_recovery_success = bool(
            p_half_rel_error <= float(case.p_half_relative_tolerance)
            and perfusion_rel_error <= float(case.perfusion_relative_tolerance)
        )
        identifiability = str(fit.get("uncertainty", {}).get("identifiability", "unknown"))
        fit_warning = bool(fit.get("fit_warning", False))
        status = "pass" if reconstruction_success and parameter_recovery_success else "review"

        if status == "pass":
            status_reason = "good observable recovery"
            if identifiability == "weak":
                status_reason += "; parameters are still only weakly separable in this regime"
        else:
            reason_parts: list[str] = []
            if not reconstruction_success:
                reason_parts.append("sensor or venous mismatch exceeded tolerance")
            if not parameter_recovery_success:
                if identifiability == "weak":
                    reason_parts.append("parameter recovery exceeded tolerance because multiple parameter combinations fit similarly well")
                else:
                    reason_parts.append("parameter recovery exceeded tolerance")
            if fit_warning:
                reason_parts.append("best-fit solution is flagged as a compromise")
            status_reason = "; ".join(reason_parts) or "review required"

        return {
            "case_id": case.case_id,
            "label": case.label,
            "category": case.category,
            "status": status,
            "status_reason": status_reason,
            "predicted_state": str(diagnostic_result.get("predicted_state", "unknown")),
            "alert_level": str(diagnostic_result.get("alert_level", "unknown")),
            "risk_score": float(diagnostic_result.get("risk_score", np.nan)),
            "truth_sensor_po2": float(truth["PO2_sensor_avg"]),
            "observed_sensor_po2": float(observed["sensor_observed"]),
            "truth_venous_sat": float(truth["S_v_percent"]),
            "observed_venous_sat": float(observed["venous_sat_observed"]),
            "truth_p_half": float(case.P_half),
            "fit_p_half": float(fit["P_half_fit"]),
            "truth_perfusion": float(case.perfusion_factor),
            "fit_perfusion": float(fit["perfusion_factor"]),
            "sensor_error": float(fit["sensor_error"]),
            "venous_error": float(fit["venous_error"]),
            "p_half_relative_error": float(p_half_rel_error),
            "perfusion_relative_error": float(perfusion_rel_error),
            "reconstruction_success": reconstruction_success,
            "parameter_recovery_success": parameter_recovery_success,
            "fit_warning": fit_warning,
            "uncertainty_summary": str(fit.get("uncertainty", {}).get("summary", "")),
            "identifiability": identifiability,
            "identifiability_summary": str(fit.get("uncertainty", {}).get("identifiability_summary", "")),
        }

    def run_suite(
        self,
        cases: list[SyntheticValidationCase] | None = None,
        *,
        sensor_noise_std: float = 0.0,
        venous_sat_noise_std: float = 0.0,
        bootstrap_samples: int = 8,
        verbose: bool = False,
    ) -> dict[str, Any]:
        selected_cases = list(cases) if cases is not None else self.cases
        rows = []
        total = len(selected_cases)
        for index, case in enumerate(selected_cases, start=1):
            if verbose:
                print(f"[validation] case {index}/{total}: {case.case_id}")
            row = self.evaluate_case(
                case,
                sensor_noise_std=sensor_noise_std,
                venous_sat_noise_std=venous_sat_noise_std,
                bootstrap_samples=bootstrap_samples,
            )
            rows.append(row)
            if verbose:
                print(
                    f"[validation]   -> {row['status']} | sensor err {row['sensor_error']:.2f} mmHg | "
                    f"venous err {row['venous_error']:.2f} mmHg"
                )

        case_count = len(rows)
        pass_count = sum(1 for row in rows if row["status"] == "pass")
        reconstruction_success_count = sum(1 for row in rows if row["reconstruction_success"])
        summary = {
            "case_count": int(case_count),
            "pass_count": int(pass_count),
            "review_count": int(case_count - pass_count),
            "reconstruction_success_rate": float(reconstruction_success_count / max(case_count, 1)),
            "mean_sensor_error": float(np.mean([row["sensor_error"] for row in rows])) if rows else np.nan,
            "mean_venous_error": float(np.mean([row["venous_error"] for row in rows])) if rows else np.nan,
            "mean_p_half_relative_error": float(np.mean([row["p_half_relative_error"] for row in rows])) if rows else np.nan,
            "mean_perfusion_relative_error": float(np.mean([row["perfusion_relative_error"] for row in rows])) if rows else np.nan,
            "max_sensor_error": float(np.max([row["sensor_error"] for row in rows])) if rows else np.nan,
            "max_venous_error": float(np.max([row["venous_error"] for row in rows])) if rows else np.nan,
        }
        return {"summary": summary, "cases": rows}

    def run_default_suite(
        self,
        *,
        sensor_noise_std: float = 0.0,
        venous_sat_noise_std: float = 0.0,
        bootstrap_samples: int = 8,
        verbose: bool = False,
    ) -> dict[str, Any]:
        return self.run_suite(
            self.cases,
            sensor_noise_std=sensor_noise_std,
            venous_sat_noise_std=venous_sat_noise_std,
            bootstrap_samples=bootstrap_samples,
            verbose=verbose,
        )

    def run_noise_robustness(
        self,
        cases: list[SyntheticValidationCase] | None = None,
        *,
        sensor_noise_levels: tuple[float, ...] = (0.0, 0.5, 1.0),
        venous_sat_noise_levels: tuple[float, ...] = (0.0, 1.0),
        bootstrap_samples: int = 4,
    ) -> dict[str, Any]:
        selected_cases = list(cases) if cases is not None else self.cases[:2]
        rows: list[dict[str, Any]] = []
        for sensor_noise in sensor_noise_levels:
            for venous_noise in venous_sat_noise_levels:
                suite = self.run_suite(
                    selected_cases,
                    sensor_noise_std=float(sensor_noise),
                    venous_sat_noise_std=float(venous_noise),
                    bootstrap_samples=int(bootstrap_samples),
                )
                rows.append({
                    "sensor_noise_std": float(sensor_noise),
                    "venous_sat_noise_std": float(venous_noise),
                    **suite["summary"],
                })

        baseline_sensor_error = rows[0]["mean_sensor_error"] if rows else np.nan
        worst_sensor_error = max((row["mean_sensor_error"] for row in rows), default=np.nan)
        summary = {
            "scenario_count": int(len(rows)),
            "baseline_sensor_error": float(baseline_sensor_error) if rows else np.nan,
            "worst_sensor_error": float(worst_sensor_error) if rows else np.nan,
            "max_error_increase": float(worst_sensor_error - baseline_sensor_error) if rows else np.nan,
        }
        return {"summary": summary, "scenarios": rows}

    def run_trend_checks(self) -> dict[str, Any]:
        base_case = self.cases[0] if self.cases else SyntheticValidationCase(
            case_id="fallback",
            label="Fallback",
            category="normoxia",
            P_inlet=90.0,
            P_half=1.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
            perfusion_factor=1.0,
        )

        perf_values = [0.5, 1.0, 2.0]
        perf_rows = []
        for perf in perf_values:
            result = self.run_single_case(
                P_inlet=float(base_case.P_inlet),
                P_half=float(base_case.P_half),
                pH=float(base_case.pH),
                pCO2=float(base_case.pCO2),
                temp_c=float(base_case.temp_c),
                perfusion_factor=float(perf),
                include_axial_diffusion=bool(base_case.include_axial_diffusion),
            )
            perf_rows.append({
                "check_type": "perfusion_trend",
                "input_value": float(perf),
                "sensor_po2": float(result["PO2_sensor_avg"]),
                "venous_po2": float(result["P_venous"]),
            })

        inlet_values = [55.0, 75.0, 95.0]
        inlet_rows = []
        for inlet in inlet_values:
            result = self.run_single_case(
                P_inlet=float(inlet),
                P_half=float(base_case.P_half),
                pH=float(base_case.pH),
                pCO2=float(base_case.pCO2),
                temp_c=float(base_case.temp_c),
                perfusion_factor=float(base_case.perfusion_factor),
                include_axial_diffusion=bool(base_case.include_axial_diffusion),
            )
            inlet_rows.append({
                "check_type": "inlet_trend",
                "input_value": float(inlet),
                "sensor_po2": float(result["PO2_sensor_avg"]),
                "venous_po2": float(result["P_venous"]),
            })

        perf_sensor_values = [row["sensor_po2"] for row in perf_rows]
        inlet_sensor_values = [row["sensor_po2"] for row in inlet_rows]
        perf_monotonic = bool(all(curr >= prev - 1e-6 for prev, curr in zip(perf_sensor_values, perf_sensor_values[1:])))
        inlet_monotonic = bool(all(curr >= prev - 1e-6 for prev, curr in zip(inlet_sensor_values, inlet_sensor_values[1:])))

        return {
            "summary": {
                "check_count": 2,
                "pass_count": int(perf_monotonic) + int(inlet_monotonic),
                "perfusion_sensor_monotonic": perf_monotonic,
                "inlet_sensor_monotonic": inlet_monotonic,
            },
            "checks": perf_rows + inlet_rows,
        }

    def save_csv_report(self, results: dict[str, Any], path: str, row_key: str = "cases") -> None:
        rows = list(results.get(row_key, []))
        if not rows:
            raise ValueError("No validation rows available to write.")

        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_json_report(self, results: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def format_summary_text(self, results: dict[str, Any]) -> str:
        summary = dict(results.get("summary", {}))
        return (
            f"Synthetic validation summary: {summary.get('pass_count', 0)}/{summary.get('case_count', 0)} cases passed; "
            f"mean sensor error {summary.get('mean_sensor_error', float('nan')):.2f} mmHg; "
            f"mean venous error {summary.get('mean_venous_error', float('nan')):.2f} mmHg; "
            f"reconstruction success rate {100.0 * summary.get('reconstruction_success_rate', 0.0):.1f}%."
        )


def run_and_save_default_validation(output_dir: str) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    runner = SyntheticValidationRunner()

    print("[validation] Running baseline synthetic suite...")
    results = runner.run_default_suite(
        sensor_noise_std=0.5,
        venous_sat_noise_std=1.0,
        bootstrap_samples=8,
        verbose=True,
    )
    runner.save_csv_report(results, os.path.join(output_dir, "synthetic_validation_report.csv"))
    runner.save_json_report(results, os.path.join(output_dir, "synthetic_validation_report.json"))

    print("[validation] Running monotonic trend checks...")
    trend_checks = runner.run_trend_checks()
    runner.save_csv_report(trend_checks, os.path.join(output_dir, "synthetic_validation_trend_checks.csv"), row_key="checks")
    runner.save_json_report(trend_checks, os.path.join(output_dir, "synthetic_validation_trend_checks.json"))

    print("[validation] Running compact robustness checks...")
    robustness = runner.run_noise_robustness(
        cases=build_default_synthetic_cases()[:1],
        sensor_noise_levels=(0.0, 1.0),
        venous_sat_noise_levels=(0.0,),
        bootstrap_samples=2,
    )
    runner.save_csv_report(robustness, os.path.join(output_dir, "synthetic_validation_robustness.csv"), row_key="scenarios")
    runner.save_json_report(robustness, os.path.join(output_dir, "synthetic_validation_robustness.json"))
    print("[validation] Reports saved.")
    return results


__all__ = [
    "SyntheticValidationCase",
    "SyntheticValidationRunner",
    "build_default_synthetic_cases",
    "build_extended_synthetic_cases",
    "run_and_save_default_validation",
]


if __name__ == "__main__":
    destination = os.path.join(os.getcwd(), "Diagnostic reports")
    report = run_and_save_default_validation(destination)
    print(SyntheticValidationRunner().format_summary_text(report))
