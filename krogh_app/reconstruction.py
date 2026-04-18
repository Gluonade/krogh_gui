"""Reconstruction service layer for Krogh-model inverse fitting."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq


class KroghReconstructor:
    """Encapsulates inverse-fitting logic used by the diagnostic reconstruction workflow."""

    def __init__(
        self,
        solve_axial_capillary_po2: Callable[..., Any],
        effective_p50: Callable[..., float],
        radial_weights: Any,
        r_vec: Any,
        z_eval: Any,
        R_cap: float,
        R_tis: float,
        L_cap: float,
    ) -> None:
        self.solve_axial_capillary_po2 = solve_axial_capillary_po2
        self.effective_p50 = effective_p50
        self.radial_weights = radial_weights
        self.r_vec = r_vec
        self.z_eval = z_eval
        self.R_cap = R_cap
        self.R_tis = R_tis
        self.L_cap = L_cap

    def _estimate_uncertainty(self, candidates: list[dict[str, Any]], best_objective: float) -> dict[str, Any]:
        """Summarize a practical near-optimal parameter band.

        This is intentionally lightweight: it does not claim a formal Bayesian or
        identifiability analysis, but it gives the user a transparent sense of how
        wide the locally plausible parameter region is.
        """
        if not candidates:
            return {
                "candidate_count": 0,
                "p_half_low": np.nan,
                "p_half_high": np.nan,
                "perfusion_low": np.nan,
                "perfusion_high": np.nan,
                "objective_threshold": np.nan,
                "wide_interval": True,
                "summary": "uncertainty band unavailable",
            }

        ranked = sorted(candidates, key=lambda item: float(item["objective"]))
        threshold = float(best_objective) + max(0.75, 0.25 * max(float(best_objective), 1.0))
        near_optimal = [item for item in ranked if float(item["objective"]) <= threshold]
        if len(near_optimal) < min(5, len(ranked)):
            near_optimal = ranked[: min(5, len(ranked))]

        p_half_values = np.array([float(item["P_half_fit"]) for item in near_optimal], dtype=float)
        perf_values = np.array([float(item["perfusion_factor"]) for item in near_optimal], dtype=float)

        p_half_low = float(np.min(p_half_values))
        p_half_high = float(np.max(p_half_values))
        perfusion_low = float(np.min(perf_values))
        perfusion_high = float(np.max(perf_values))

        wide_interval = bool(
            (p_half_high / max(p_half_low, 1e-6) > 4.0)
            or (perfusion_high / max(perfusion_low, 1e-6) > 2.5)
        )

        return {
            "candidate_count": int(len(near_optimal)),
            "p_half_low": p_half_low,
            "p_half_high": p_half_high,
            "perfusion_low": perfusion_low,
            "perfusion_high": perfusion_high,
            "objective_threshold": float(threshold),
            "wide_interval": wide_interval,
            "summary": (
                f"mitoP50 ≈ {p_half_low:.3g}–{p_half_high:.3g} mmHg | "
                f"perfusion ≈ {perfusion_low:.3g}–{perfusion_high:.3g}x "
                f"({len(near_optimal)} near-optimal fits)"
            ),
        }

    def _spatial_fraction_below(self, tissue_po2: Any, threshold: float, radial_mask: Any | None = None) -> float:
        values = np.maximum(np.asarray(tissue_po2, dtype=float), 0.0)
        if values.ndim != 2 or values.size == 0:
            return 0.0

        weights = np.asarray(self.radial_weights, dtype=float)
        if radial_mask is not None:
            mask = np.asarray(radial_mask, dtype=bool)
            if mask.shape[0] != values.shape[1]:
                mask = np.ones(values.shape[1], dtype=bool)
            if not np.any(mask):
                mask = np.zeros(values.shape[1], dtype=bool)
                mask[0] = True
            values = values[:, mask]
            weights = weights[mask]
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones(values.shape[1], dtype=float)

        below_mask = (values < float(threshold)).astype(float)
        radial_fraction = np.average(below_mask, axis=1, weights=weights)
        return float(np.mean(radial_fraction))

    def _radius_scenario_mask(self, radius_um: float) -> np.ndarray:
        radius_limit = min(float(self.R_tis), max(float(self.R_cap), float(radius_um) * 1e-4))
        radial_positions = np.asarray(self.r_vec, dtype=float)
        mask = radial_positions <= (radius_limit + 1e-12)
        if not np.any(mask):
            mask = np.zeros_like(radial_positions, dtype=bool)
            mask[0] = True
        return mask

    def _classify_radius_alert(self, scenario: dict[str, Any]) -> tuple[str, str]:
        sensor_avg = float(scenario.get("sensor_avg", 0.0))
        below_1 = float(scenario.get("fraction_below_1", 0.0))
        below_5 = float(scenario.get("fraction_below_5", 0.0))
        below_10 = float(scenario.get("fraction_below_10", 0.0))
        below_15 = float(scenario.get("fraction_below_15", 0.0))

        if below_1 > 0.01 or below_5 >= 0.10 or sensor_avg < 15.0:
            return "red", "serious risk of severe hypoxia"
        if below_5 >= 0.03 or below_10 >= 0.20 or sensor_avg < 18.0:
            return "orange", "high probability of major hypoxic spots"
        if below_10 >= 0.08 or below_15 >= 0.20 or sensor_avg < 23.0:
            return "yellow", "increased alert with a relevant hidden hypoxic burden"
        return "green", "no major hidden hypoxic burden suggested under this radius assumption"

    def _build_radius_scenarios(self, tissue_po2: Any) -> tuple[dict[str, dict[str, Any]], str]:
        scenario_defs = (
            ("normal_30um", "normal", 30.0),
            ("increased_50um", "increased", 50.0),
            ("high_100um", "high", 100.0),
        )
        values = np.maximum(np.asarray(tissue_po2, dtype=float), 0.0)
        scenarios: dict[str, dict[str, Any]] = {}

        for key, label, radius_um in scenario_defs:
            mask = self._radius_scenario_mask(radius_um)
            subset = values[:, mask]
            weights = np.asarray(self.radial_weights, dtype=float)[mask]
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones(subset.shape[1], dtype=float)
            mean_profile = np.average(subset, axis=1, weights=weights)
            scenario = {
                "label": label,
                "radius_um": float(radius_um),
                "sensor_avg": float(np.mean(mean_profile)),
                "fraction_below_1": self._spatial_fraction_below(values, 1.0, radial_mask=mask),
                "fraction_below_5": self._spatial_fraction_below(values, 5.0, radial_mask=mask),
                "fraction_below_10": self._spatial_fraction_below(values, 10.0, radial_mask=mask),
                "fraction_below_15": self._spatial_fraction_below(values, 15.0, radial_mask=mask),
            }
            scenario["alert_level"], scenario["interpretation"] = self._classify_radius_alert(scenario)
            scenarios[key] = scenario

        summary = (
            "Radius sensitivity across assumed tissue radius: "
            f"30 µm -> {scenarios['normal_30um']['alert_level']} "
            f"(mean tissue PO2 {scenarios['normal_30um']['sensor_avg']:.1f} mmHg, "
            f"{100.0 * scenarios['normal_30um']['fraction_below_10']:.1f}% below 10 mmHg); "
            f"50 µm -> {scenarios['increased_50um']['alert_level']} "
            f"(mean tissue PO2 {scenarios['increased_50um']['sensor_avg']:.1f} mmHg, "
            f"{100.0 * scenarios['increased_50um']['fraction_below_10']:.1f}% below 10 mmHg); "
            f"100 µm -> {scenarios['high_100um']['alert_level']} "
            f"(mean tissue PO2 {scenarios['high_100um']['sensor_avg']:.1f} mmHg, "
            f"{100.0 * scenarios['high_100um']['fraction_below_10']:.1f}% below 10 mmHg)."
        )
        if scenarios['high_100um']['alert_level'] != scenarios['normal_30um']['alert_level']:
            summary += " In this case, the elevated concern appears mainly under the larger, swollen-tissue radius assumption."
        return scenarios, summary

    def _build_hypoxic_burden_summary(self, fraction_map: dict[str, float]) -> str:
        return (
            "Estimated hidden hypoxic burden: "
            f"{100.0 * fraction_map['below_10']:.1f}% of the tissue cylinder is below 10 mmHg, "
            f"{100.0 * fraction_map['below_5']:.1f}% is below 5 mmHg, and "
            f"{100.0 * fraction_map['below_1']:.1f}% is below 1 mmHg."
        )

    def _build_assumption_summary(
        self,
        *,
        perfusion_factor: float,
        p_half_fit: float,
        sensor_target: float,
        sensor_sim: float,
        venous_target: float,
        venous_sim: float,
        fit_warning: bool,
    ) -> str:
        if perfusion_factor < 0.9:
            perfusion_phrase = f"reduced effective perfusion near {perfusion_factor:.2f} x baseline"
        elif perfusion_factor > 1.1:
            perfusion_phrase = f"increased effective perfusion near {perfusion_factor:.2f} x baseline"
        else:
            perfusion_phrase = f"effective perfusion close to baseline ({perfusion_factor:.2f} x)"

        text = (
            "To reproduce the measured arterial, tissue-sensor, and venous pattern, "
            f"the best fit required {perfusion_phrase} and mitoP50 near {p_half_fit:.2f} mmHg. "
            f"The base Krogh geometry used a tissue cylinder radius of {self.R_tis * 1e4:.0f} µm. "
            f"The fitted cylinder reproduced sensor PO2 as {sensor_sim:.1f} mmHg for a target of {sensor_target:.1f} mmHg, "
            f"and venous PO2 as {venous_sim:.1f} mmHg for a target of {venous_target:.1f} mmHg."
        )
        if fit_warning:
            text += " The match should be treated as a best available compromise rather than a unique explanation."
        return text

    def fit_p_half_from_venous(
        self,
        P_inlet: float,
        P_v_target: float,
        pH: float,
        pCO2: float,
        temp_c: float,
        perfusion_factor: float = 1.0,
        include_axial: bool = True,
    ) -> tuple[float, float]:
        p50_eff = self.effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)

        def _simulate_venous(P_half_candidate: float) -> float:
            P_c_axial, _, _ = self.solve_axial_capillary_po2(
                P_inlet=P_inlet,
                P_half=P_half_candidate,
                p50_eff=p50_eff,
                include_axial_diffusion=include_axial,
                perfusion_factor=perfusion_factor,
            )
            return float(P_c_axial[-1])

        def _residual(P_half_candidate: float) -> float:
            return _simulate_venous(P_half_candidate) - P_v_target

        lo, hi = 0.05, 100.0
        try:
            f_lo = _residual(lo)
            f_hi = _residual(hi)
            if f_lo * f_hi > 0:
                P_half_fit = lo if abs(f_lo) <= abs(f_hi) else hi
            else:
                P_half_fit = brentq(_residual, lo, hi, xtol=1e-4, rtol=1e-4, maxiter=80)
        except Exception:
            P_half_fit = 1.0

        return float(P_half_fit), float(_simulate_venous(P_half_fit))

    def fit_joint_parameters(
        self,
        P_inlet: float,
        sensor_target: float,
        P_v_target: float,
        pH: float,
        pCO2: float,
        temp_c: float,
        include_axial: bool = True,
        venous_weight: float = 0.15,
    ) -> dict[str, Any]:
        p50_eff = self.effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)
        perf_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.35, 5.0, 15))))
        p_half_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.05, 300.0, 22))))
        best = None
        evaluated_candidates: list[dict[str, Any]] = []

        for perf in perf_candidates:
            for p_half in p_half_candidates:
                try:
                    P_c_axial, tissue_po2, _ = self.solve_axial_capillary_po2(
                        P_inlet=P_inlet,
                        P_half=float(p_half),
                        p50_eff=p50_eff,
                        include_axial_diffusion=include_axial,
                        perfusion_factor=float(perf),
                    )
                    tissue_po2 = np.maximum(tissue_po2, 0.0)
                    sensor_sim = float(np.average(tissue_po2, axis=1, weights=self.radial_weights).mean())
                    P_v_sim = float(P_c_axial[-1])
                    sensor_error = abs(sensor_sim - sensor_target)
                    venous_error = abs(P_v_sim - P_v_target)
                    objective = (
                        sensor_error
                        + float(venous_weight) * venous_error
                        + 0.10 * abs(np.log(float(perf)))
                        + 0.02 * abs(np.log(max(float(p_half), 1e-6)))
                    )

                    candidate = {
                        "objective": float(objective),
                        "perfusion_factor": float(perf),
                        "P_half_fit": float(p_half),
                        "P_v_sim": float(P_v_sim),
                        "sensor_sim": float(sensor_sim),
                        "sensor_error": float(sensor_error),
                        "venous_error": float(venous_error),
                        "p50_eff": float(p50_eff),
                        "P_c_axial": P_c_axial,
                        "tissue_po2": tissue_po2,
                    }
                    evaluated_candidates.append(candidate)
                    if best is None or candidate["objective"] < best["objective"]:
                        best = candidate
                except Exception:
                    continue

        if best is None:
            raise RuntimeError("Could not fit Krogh parameters for the selected diagnostic inputs.")

        best["uncertainty"] = self._estimate_uncertainty(evaluated_candidates, float(best["objective"]))
        best["fit_warning"] = bool(
            best["sensor_error"] > 6.0
            or (float(venous_weight) >= 0.4 and best["venous_error"] > 4.0)
            or bool(best["uncertainty"].get("wide_interval", False))
        )
        return best

    def build_plot_data(
        self,
        *,
        po2: float,
        pco2: float,
        ph: float,
        temperature_c: float,
        venous_sat: float,
        P_v_target: float,
        venous_weight: float,
        sensor_po2: float,
        diag_result: dict[str, Any],
        fit: dict[str, Any],
    ) -> dict[str, Any]:
        P_half_fit = float(fit["P_half_fit"])
        P_v_sim = float(fit["P_v_sim"])
        p50_eff = float(fit["p50_eff"])
        perfusion_factor = float(fit["perfusion_factor"])
        P_c_axial = fit["P_c_axial"]
        tissue_po2 = np.maximum(fit["tissue_po2"], 0.0)

        P_avg = np.average(tissue_po2, axis=1, weights=self.radial_weights)
        hypoxic_fraction_map = {
            "below_1": self._spatial_fraction_below(tissue_po2, 1.0),
            "below_5": self._spatial_fraction_below(tissue_po2, 5.0),
            "below_10": self._spatial_fraction_below(tissue_po2, 10.0),
            "below_15": self._spatial_fraction_below(tissue_po2, 15.0),
            "below_20": self._spatial_fraction_below(tissue_po2, 20.0),
        }
        hypoxic_burden_summary = self._build_hypoxic_burden_summary(hypoxic_fraction_map)
        radius_scenarios, radius_sensitivity_summary = self._build_radius_scenarios(tissue_po2)
        assumption_summary = self._build_assumption_summary(
            perfusion_factor=perfusion_factor,
            p_half_fit=P_half_fit,
            sensor_target=float(sensor_po2),
            sensor_sim=float(fit["sensor_sim"]),
            venous_target=float(P_v_target),
            venous_sim=P_v_sim,
            fit_warning=bool(fit["fit_warning"]),
        )
        x_sym = np.linspace(-self.R_tis, self.R_tis, 2 * len(self.r_vec) - 1)
        X_sym, Z_sym = np.meshgrid(x_sym, self.z_eval, indexing="xy")
        Z_rel = Z_sym / self.L_cap
        R_abs = np.abs(x_sym)
        PO2_sym = np.zeros((len(self.z_eval), len(x_sym)), dtype=float)
        for i, Pc in enumerate(P_c_axial):
            PO2_sym[i, :] = np.interp(R_abs, self.r_vec, tissue_po2[i, :], left=Pc, right=tissue_po2[i, -1])
        PO2_sym = np.where(R_abs[None, :] < self.R_cap, P_c_axial[:, None], PO2_sym)
        PO2_sym = np.maximum(PO2_sym, 0.0)

        return {
            "po2": po2,
            "pco2": pco2,
            "ph": ph,
            "temperature_c": temperature_c,
            "p50_eff": p50_eff,
            "P_half_fit": P_half_fit,
            "P_v_target": P_v_target,
            "P_v_sim": P_v_sim,
            "venous_sat": venous_sat,
            "venous_weight": float(venous_weight),
            "perfusion_factor": perfusion_factor,
            "sensor_target": float(sensor_po2),
            "sensor_error": float(fit["sensor_error"]),
            "venous_error": float(fit["venous_error"]),
            "fit_warning": bool(fit["fit_warning"]),
            "uncertainty": dict(fit.get("uncertainty", {})),
            "hypoxic_fraction_map": hypoxic_fraction_map,
            "hypoxic_burden_summary": hypoxic_burden_summary,
            "radius_scenarios": radius_scenarios,
            "radius_sensitivity_summary": radius_sensitivity_summary,
            "assumption_summary": assumption_summary,
            "diag_result": diag_result,
            "P_c_axial": P_c_axial,
            "P_avg": P_avg,
            "X_sym": X_sym,
            "Z_rel": Z_rel,
            "PO2_sym": PO2_sym,
            "po2_min_plot": 0.0,
            "po2_max_plot": max(float(po2) * 1.05, 30.0),
            "p_venous": float(P_c_axial[-1]),
            "p_tis_min": float(np.min(tissue_po2)),
            "sensor_avg": float(P_avg.mean()),
            "sensor_sim": float(fit["sensor_sim"]),
        }


__all__ = ["KroghReconstructor"]
