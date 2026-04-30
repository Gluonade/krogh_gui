"""Series and sweep service layer for Krogh-model studies."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


class SeriesRunner:
    """Encapsulates sweep creation, case construction, and numerical check logic."""

    def __init__(
        self,
        run_single_case: Callable[..., dict[str, Any]],
        series_sweep_fields: dict[str, str],
        build_tighter_numeric_settings: Callable[[dict[str, Any]], dict[str, Any]],
        build_series_check_indices: Callable[[int], list[int]],
        temporary_numeric_settings: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.run_single_case = run_single_case
        self.series_sweep_fields = series_sweep_fields
        self.build_tighter_numeric_settings = build_tighter_numeric_settings
        self.build_series_check_indices = build_series_check_indices
        self.temporary_numeric_settings = temporary_numeric_settings

    def build_values(self, start_value: float, end_value: float, step_size: float):
        start_value = float(start_value)
        end_value = float(end_value)
        step_size = float(step_size)

        if step_size <= 0.0:
            raise ValueError("Step size must be > 0.")

        distance = end_value - start_value
        if np.isclose(distance, 0.0):
            return np.array([start_value], dtype=float)

        direction = 1.0 if distance > 0.0 else -1.0
        signed_step = direction * step_size
        sweep_values = np.arange(start_value, end_value + 0.5 * signed_step, signed_step, dtype=float)

        if direction > 0.0:
            sweep_values = sweep_values[sweep_values <= end_value]
        else:
            sweep_values = sweep_values[sweep_values >= end_value]

        if sweep_values.size == 0 or not np.isclose(sweep_values[-1], end_value):
            sweep_values = np.append(sweep_values, end_value)

        return sweep_values

    def build_case_definitions(
        self,
        base_params: dict[str, Any],
        sweep_field_label: str,
        sweep_values,
        secondary_field_label: str | None = None,
        secondary_values=None,
    ) -> list[dict[str, Any]]:
        if sweep_field_label not in self.series_sweep_fields:
            raise ValueError(f"Unsupported sweep parameter: {sweep_field_label}")
        if secondary_field_label is not None and secondary_field_label not in self.series_sweep_fields:
            raise ValueError(f"Unsupported secondary sweep parameter: {secondary_field_label}")

        primary_param_name = self.series_sweep_fields[sweep_field_label]
        secondary_param_name = self.series_sweep_fields.get(secondary_field_label)
        second_values = [None] if secondary_field_label is None else list(secondary_values)
        case_definitions = []
        case_index = 1

        for secondary_value in second_values:
            for sweep_value in sweep_values:
                case_params = dict(base_params)
                case_params[primary_param_name] = float(sweep_value)
                if secondary_param_name is not None:
                    case_params[secondary_param_name] = float(secondary_value)
                if case_params["perf"] <= 0.0:
                    raise ValueError("Perfusion factor must be > 0 for all series values.")
                case_definitions.append(
                    {
                        "case_index": case_index,
                        "sweep_field_label": sweep_field_label,
                        "sweep_value": float(sweep_value),
                        "secondary_field_label": secondary_field_label,
                        "secondary_sweep_value": None if secondary_value is None else float(secondary_value),
                        "case_params": case_params,
                    }
                )
                case_index += 1

        return case_definitions

    def build_result_row(
        self,
        case_index: int,
        sweep_field_label: str,
        sweep_value: float,
        case_params: dict[str, Any],
        secondary_field_label: str | None = None,
        secondary_sweep_value: float | None = None,
    ) -> dict[str, Any]:
        result = self.run_single_case(
            P_inlet=case_params["P_inlet"],
            P_half=case_params["P_half"],
            pH=case_params["pH"],
            pCO2=case_params["pCO2"],
            temp_c=case_params["temp_c"],
            perfusion_factor=case_params["perf"],
            metabolic_rate_rel=case_params.get("metabolic_rate_rel", 1.0),
            include_axial_diffusion=case_params["include_axial"],
            high_po2_threshold_primary=case_params["high_po2_threshold_primary"],
            high_po2_threshold_secondary=case_params["high_po2_threshold_secondary"],
            additional_high_po2_thresholds=case_params.get("additional_high_po2_thresholds", []),
            relative_high_po2_thresholds_percent=case_params.get("relative_high_po2_thresholds_percent", [90.0, 50.0, 30.0]),
            relative_high_po2_reference=case_params.get("relative_high_po2_reference", "inlet"),
        )
        return {
            "Case": case_index,
            "Sweep_parameter": sweep_field_label,
            "Sweep_value": float(sweep_value),
            "Sweep_parameter_2": secondary_field_label or "",
            "Sweep_value_2": np.nan if secondary_sweep_value is None else float(secondary_sweep_value),
            "PO2_inlet_mmHg": float(case_params["P_inlet"]),
            "mitoP50_mmHg": float(case_params["P_half"]),
            "pH": float(case_params["pH"]),
            "pCO2_mmHg": float(case_params["pCO2"]),
            "Temp_C": float(case_params["temp_c"]),
            "Perfusion_factor": float(case_params["perf"]),
            "Metabolic_rate_rel": float(case_params.get("metabolic_rate_rel", 1.0)),
            "Tissue_radius_um": float(case_params.get("tissue_radius_um", 100.0)),
            "High_PO2_threshold_1_mmHg": float(case_params["high_po2_threshold_primary"]),
            "High_PO2_threshold_2_mmHg": float(case_params["high_po2_threshold_secondary"]),
            "High_PO2_additional_thresholds_mmHg": ", ".join(f"{value:.6g}" for value in case_params.get("additional_high_po2_thresholds", [])),
            "High_PO2_relative_thresholds_percent": ", ".join(f"{value:.6g}" for value in case_params.get("relative_high_po2_thresholds_percent", [])),
            "Relative_PO2_reference": str(case_params.get("relative_high_po2_reference", "inlet")),
            "Include_axial_diffusion": bool(case_params["include_axial"]),
            **result,
        }

    def build_result_row_from_definition(self, case_definition: dict[str, Any]) -> dict[str, Any]:
        return self.build_result_row(
            case_definition["case_index"],
            case_definition["sweep_field_label"],
            case_definition["sweep_value"],
            case_definition["case_params"],
            secondary_field_label=case_definition.get("secondary_field_label"),
            secondary_sweep_value=case_definition.get("secondary_sweep_value"),
        )

    def run_case_definitions(self, case_definitions: list[dict[str, Any]], numeric_settings: dict[str, Any], per_case_callback=None):
        with self.temporary_numeric_settings(numeric_settings):
            results = []
            for case_definition in case_definitions:
                if per_case_callback is not None:
                    per_case_callback(case_definition, len(case_definitions))
                results.append(self.build_result_row_from_definition(case_definition))
        return results

    def analyze_numerics(
        self,
        case_definitions: list[dict[str, Any]],
        results_df: pd.DataFrame,
        selected_fields: list[str],
        numeric_settings: dict[str, Any],
    ) -> dict[str, Any]:
        sample_indices = self.build_series_check_indices(len(case_definitions))
        sampled_case_numbers = [index + 1 for index in sample_indices]
        tighter_settings = self.build_tighter_numeric_settings(numeric_settings)
        check_rows = []

        with self.temporary_numeric_settings(tighter_settings):
            for index in sample_indices:
                check_rows.append(self.build_result_row_from_definition(case_definitions[index]))

        refined_df = pd.DataFrame(check_rows).set_index("Case")
        baseline_df = results_df.set_index("Case")
        field_metrics = {}
        has_warning = False

        for field_name in selected_fields:
            baseline_values = baseline_df.loc[sampled_case_numbers, field_name].to_numpy(dtype=float)
            refined_values = refined_df.loc[sampled_case_numbers, field_name].to_numpy(dtype=float)
            abs_diffs = np.abs(baseline_values - refined_values)
            scales = np.maximum(np.abs(refined_values), 1e-9)
            rel_diffs = abs_diffs / scales
            worst_index = int(np.argmax(abs_diffs)) if abs_diffs.size else 0
            max_abs_diff = float(abs_diffs[worst_index]) if abs_diffs.size else 0.0
            max_rel_diff = float(rel_diffs[worst_index]) if abs_diffs.size else 0.0
            worst_case = int(sampled_case_numbers[worst_index]) if sampled_case_numbers else 0

            field_metrics[field_name] = {
                "max_abs_diff": max_abs_diff,
                "max_rel_diff": max_rel_diff,
                "worst_case": worst_case,
            }
            if max_abs_diff > 0.05 or max_rel_diff > 0.01:
                has_warning = True

        return {
            "sample_count": len(sample_indices),
            "field_metrics": field_metrics,
            "has_warning": has_warning,
        }


__all__ = ["SeriesRunner"]
