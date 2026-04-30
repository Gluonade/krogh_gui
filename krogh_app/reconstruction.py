"""Reconstruction service layer for Krogh-model inverse fitting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq, minimize


class KroghReconstructor:
    """Encapsulates inverse-fitting logic used by the diagnostic reconstruction workflow."""

    def _call_solver(
        self,
        *,
        P_inlet: float,
        P_half: float,
        p50_eff: float,
        include_axial: bool,
        perfusion_factor: float,
        metabolic_rate_rel: float,
    ) -> tuple[Any, Any, Any]:
        try:
            return self.solve_axial_capillary_po2(
                P_inlet=P_inlet,
                P_half=P_half,
                p50_eff=p50_eff,
                include_axial_diffusion=include_axial,
                perfusion_factor=perfusion_factor,
                metabolic_rate_rel=metabolic_rate_rel,
            )
        except TypeError as exc:
            if "metabolic_rate_rel" not in str(exc):
                raise
            return self.solve_axial_capillary_po2(
                P_inlet=P_inlet,
                P_half=P_half,
                p50_eff=p50_eff,
                include_axial_diffusion=include_axial,
                perfusion_factor=perfusion_factor,
            )

    def _build_fast_numeric_settings(self, base_settings: dict[str, float | int]) -> dict[str, float | int]:
        if self.build_fast_numeric_settings is not None:
            return dict(self.build_fast_numeric_settings(base_settings))

        return {
            "ode_rtol": min(max(float(base_settings["ode_rtol"]) * 25.0, 1e-10), 1e-4),
            "ode_atol": min(max(float(base_settings["ode_atol"]) * 25.0, 1e-12), 1e-6),
            "ode_max_step": max(float(base_settings["ode_max_step"]) * 1.5, 1e-6),
            "axial_diffusion_max_iter": max(20, int(round(int(base_settings["axial_diffusion_max_iter"]) * 0.35))),
            "axial_diffusion_tol": min(max(float(base_settings["axial_diffusion_tol"]) * 30.0, 1e-10), 1e-2),
            "axial_coupling_max_iter": max(2, int(round(int(base_settings["axial_coupling_max_iter"]) * 0.5))),
            "axial_coupling_tol": min(max(float(base_settings["axial_coupling_tol"]) * 30.0, 1e-10), 1e-2),
            "bootstrap_samples": int(base_settings.get("bootstrap_samples", 0)),
        }

    def _temporary_numeric_context(self, *, use_fast_mode: bool):
        if not use_fast_mode or self.get_numeric_settings is None or self.temporary_numeric_settings is None:
            return nullcontext()

        return self.temporary_numeric_settings(self._build_fast_numeric_settings(self.get_numeric_settings()))

    def _evaluate_single_candidate(
        self,
        *,
        perfusion_factor: float,
        p_half: float,
        metabolic_rate_rel: float,
        P_inlet: float,
        p50_eff: float,
        include_axial: bool,
        sensor_target: float,
        P_v_target: float,
        metabolic_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
        solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any] | None:
        cache_key = (
            round(float(perfusion_factor), 10),
            round(float(p_half), 10),
            round(float(metabolic_rate_rel), 10),
            bool(include_axial),
        )
        if solver_cache is not None and cache_key in solver_cache:
            return solver_cache[cache_key]

        candidate: dict[str, Any] | None = None
        try:
            P_c_axial, tissue_po2, _ = self._call_solver(
                P_inlet=float(P_inlet),
                P_half=float(p_half),
                p50_eff=float(p50_eff),
                include_axial=bool(include_axial),
                perfusion_factor=float(perfusion_factor),
                metabolic_rate_rel=float(metabolic_rate_rel),
            )
            candidate = self._candidate_from_solution(
                P_c_axial=P_c_axial,
                tissue_po2=tissue_po2,
                perfusion_factor=float(perfusion_factor),
                p_half=float(p_half),
                metabolic_rate_rel=float(metabolic_rate_rel),
                metabolic_target=float(metabolic_target),
                p50_eff=float(p50_eff),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
            )
        except Exception:
            candidate = None

        if solver_cache is not None:
            solver_cache[cache_key] = candidate
        return candidate

    def _candidate_from_solution(
        self,
        *,
        P_c_axial: Any,
        tissue_po2: Any,
        perfusion_factor: float,
        p_half: float,
        metabolic_rate_rel: float,
        metabolic_target: float,
        p50_eff: float,
        sensor_target: float,
        P_v_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
    ) -> dict[str, Any]:
        tissue_po2 = np.maximum(np.asarray(tissue_po2, dtype=float), 0.0)
        sensor_sim = float(np.average(tissue_po2, axis=1, weights=self.radial_weights).mean())
        P_v_sim = float(np.asarray(P_c_axial, dtype=float)[-1])
        sensor_error = abs(sensor_sim - sensor_target)
        venous_error = abs(P_v_sim - P_v_target)
        boundary_hit = bool(
            np.isclose(float(perfusion_factor), perf_lo)
            or np.isclose(float(perfusion_factor), perf_hi)
            or np.isclose(float(p_half), p_half_lo)
            or np.isclose(float(p_half), p_half_hi)
            or np.isclose(float(metabolic_rate_rel), met_lo)
            or np.isclose(float(metabolic_rate_rel), met_hi)
        )
        objective = self._objective_value(
            sensor_sim=float(sensor_sim),
            sensor_target=float(sensor_target),
            P_v_sim=float(P_v_sim),
            P_v_target=float(P_v_target),
            perfusion_factor=float(perfusion_factor),
            p_half=float(p_half),
            metabolic_rate_rel=float(metabolic_rate_rel),
            metabolic_target=float(metabolic_target),
            venous_weight=float(venous_weight),
        )

        return {
            "objective": float(objective),
            "perfusion_factor": float(perfusion_factor),
            "P_half_fit": float(p_half),
            "metabolic_rate_rel": float(metabolic_rate_rel),
            "metabolic_target": float(metabolic_target),
            "fit_boundary_hit": bool(boundary_hit),
            "P_v_sim": float(P_v_sim),
            "sensor_sim": float(sensor_sim),
            "sensor_error": float(sensor_error),
            "venous_error": float(venous_error),
            "p50_eff": float(p50_eff),
            "P_c_axial": np.asarray(P_c_axial, dtype=float),
            "tissue_po2": tissue_po2,
        }

    def _evaluate_candidate_points(
        self,
        candidate_points: list[tuple[float, float, float]],
        *,
        P_inlet: float,
        p50_eff: float,
        include_axial: bool,
        sensor_target: float,
        P_v_target: float,
        metabolic_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
        solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None] | None = None,
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        unique_points = list(dict.fromkeys((float(perf), float(p_half), float(metabolic_rate_rel)) for perf, p_half, metabolic_rate_rel in candidate_points))

        def evaluate_point(point: tuple[float, float, float]) -> dict[str, Any] | None:
            perf, p_half, metabolic_rate_rel = point
            return self._evaluate_single_candidate(
                perfusion_factor=float(perf),
                p_half=float(p_half),
                metabolic_rate_rel=float(metabolic_rate_rel),
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                include_axial=bool(include_axial),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
                solver_cache=solver_cache,
            )

        if parallel and len(unique_points) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(unique_points))) as executor:
                evaluated = list(executor.map(evaluate_point, unique_points))
        else:
            evaluated = [evaluate_point(point) for point in unique_points]

        return [candidate for candidate in evaluated if candidate is not None]

    def _select_candidate_indices(
        self,
        values: np.ndarray,
        *,
        max_points: int,
        preferred_values: tuple[float, ...] = (),
    ) -> list[int]:
        if len(values) <= max_points:
            return list(range(len(values)))

        indices = set(np.linspace(0, len(values) - 1, max_points, dtype=int).tolist())
        for preferred in preferred_values:
            indices.add(int(np.argmin(np.abs(values - float(preferred)))))
        return sorted(index for index in indices if 0 <= index < len(values))

    def _screen_candidate_points(
        self,
        *,
        perf_candidates: np.ndarray,
        p_half_candidates: np.ndarray,
        metabolic_candidates: np.ndarray,
        P_inlet: float,
        p50_eff: float,
        sensor_target: float,
        P_v_target: float,
        metabolic_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
    ) -> list[tuple[int, int, int]]:
        perf_indices = self._select_candidate_indices(
            perf_candidates,
            max_points=5,
            preferred_values=(1.0,),
        )
        p_half_indices = self._select_candidate_indices(
            p_half_candidates,
            max_points=7,
            preferred_values=(1.0,),
        )
        metabolic_indices = self._select_candidate_indices(
            metabolic_candidates,
            max_points=4,
            preferred_values=(1.0, float(metabolic_target)),
        )

        coarse_index_points = [
            (perf_index, p_half_index, metabolic_index)
            for perf_index in perf_indices
            for p_half_index in p_half_indices
            for metabolic_index in metabolic_indices
        ]
        coarse_value_points = [
            (
                float(perf_candidates[perf_index]),
                float(p_half_candidates[p_half_index]),
                float(metabolic_candidates[metabolic_index]),
            )
            for perf_index, p_half_index, metabolic_index in coarse_index_points
        ]

        with self._temporary_numeric_context(use_fast_mode=True):
            coarse_candidates = self._evaluate_candidate_points(
                coarse_value_points,
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                include_axial=False,
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
            )
        if not coarse_candidates:
            return []

        index_by_value = {
            (
                float(perf_candidates[perf_index]),
                float(p_half_candidates[p_half_index]),
                float(metabolic_candidates[metabolic_index]),
            ): (perf_index, p_half_index, metabolic_index)
            for perf_index, p_half_index, metabolic_index in coarse_index_points
        }

        top_seed_count = min(1, len(coarse_candidates))
        ranked = sorted(coarse_candidates, key=lambda item: float(item["objective"]))[:top_seed_count]
        expanded_index_points: set[tuple[int, int, int]] = set()

        for candidate in ranked:
            perf_index, p_half_index, metabolic_index = index_by_value[
                (
                    float(candidate["perfusion_factor"]),
                    float(candidate["P_half_fit"]),
                    float(candidate.get("metabolic_rate_rel", 1.0)),
                )
            ]
            perf_range = range(max(0, perf_index - 1), min(len(perf_candidates), perf_index + 2))
            p_half_range = range(max(0, p_half_index - 1), min(len(p_half_candidates), p_half_index + 2))
            metabolic_range = range(max(0, metabolic_index - 1), min(len(metabolic_candidates), metabolic_index + 2))
            for perf_candidate_index in perf_range:
                for p_half_candidate_index in p_half_range:
                    for metabolic_candidate_index in metabolic_range:
                        expanded_index_points.add(
                            (perf_candidate_index, p_half_candidate_index, metabolic_candidate_index)
                        )

        baseline_index = (
            int(np.argmin(np.abs(perf_candidates - 1.0))),
            int(np.argmin(np.abs(p_half_candidates - 1.0))),
            int(np.argmin(np.abs(metabolic_candidates - float(metabolic_target)))),
        )
        expanded_index_points.add(baseline_index)
        return sorted(expanded_index_points)

    def _objective_value(
        self,
        *,
        sensor_sim: float,
        sensor_target: float,
        P_v_sim: float,
        P_v_target: float,
        perfusion_factor: float,
        p_half: float,
        metabolic_rate_rel: float,
        metabolic_target: float,
        venous_weight: float,
    ) -> float:
        sensor_error = abs(float(sensor_sim) - float(sensor_target))
        venous_error = abs(float(P_v_sim) - float(P_v_target))
        return float(
            sensor_error
            + float(venous_weight) * venous_error
            + 0.10 * abs(np.log(max(float(perfusion_factor), 1e-6)))
            + 0.008 * abs(np.log(max(float(p_half), 1e-6)))
            + 0.12 * abs(np.log(max(float(metabolic_rate_rel), 1e-6) / max(float(metabolic_target), 1e-6)))
        )

    def _refine_candidate_to_sensor_target(
        self,
        *,
        seed_candidate: dict[str, Any],
        P_inlet: float,
        p50_eff: float,
        include_axial: bool,
        sensor_target: float,
        P_v_target: float,
        metabolic_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
        solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None],
    ) -> dict[str, Any]:
        """Two-stage 1-D brentq refinement: metabolic rate first (expanded upper bound), then perfusion.

        mitoP50 bounds are never modified. Expanding metabolic upper bound to match sensor
        mean is the primary physiological mechanism (increased O2 demand lowers tissue pO2);
        reducing perfusion is the secondary mechanism.
        """
        fixed_p_half = float(seed_candidate["P_half_fit"])
        fixed_metabolic = float(seed_candidate.get("metabolic_rate_rel", metabolic_target))
        fixed_perfusion = float(seed_candidate["perfusion_factor"])
        # Expand metabolic upper bound only (mitoP50 bounds stay unchanged)
        expanded_met_hi = max(float(met_hi) * 5.0, 12.0)

        def _eval(
            perfusion_factor: float,
            metabolic_rate_rel: float,
            use_axial: bool,
            met_hi_override: float | None = None,
        ) -> dict[str, Any] | None:
            return self._evaluate_single_candidate(
                perfusion_factor=float(perfusion_factor),
                p_half=float(fixed_p_half),
                metabolic_rate_rel=float(metabolic_rate_rel),
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                include_axial=bool(use_axial),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi_override if met_hi_override is not None else met_hi),
                solver_cache=solver_cache,
            )

        # fast (no axial) helpers used for all search evaluations
        def evaluate_met_fast(m: float) -> dict[str, Any] | None:
            return _eval(fixed_perfusion, m, use_axial=False, met_hi_override=expanded_met_hi)

        def evaluate_perf_fast(p: float) -> dict[str, Any] | None:
            return _eval(p, best_metabolic, use_axial=False)

        baseline = _eval(fixed_perfusion, fixed_metabolic, use_axial=False)
        if baseline is None:
            return seed_candidate

        best_candidate = baseline
        best_abs_error = abs(float(baseline["sensor_error"]))
        best_metabolic = fixed_metabolic  # updated by stage 1
        best_perfusion = fixed_perfusion  # updated by stage 2

        # --- Stage 1: brentq over metabolic rate (fast, no axial) ---
        met_lo_c = evaluate_met_fast(float(met_lo))
        met_hi_c = evaluate_met_fast(float(expanded_met_hi))
        for c in (met_lo_c, met_hi_c):
            if c is not None and abs(float(c["sensor_error"])) < best_abs_error:
                best_candidate = c
                best_abs_error = abs(float(c["sensor_error"]))
                best_metabolic = float(c["metabolic_rate_rel"])
                best_perfusion = float(c["perfusion_factor"])

        if met_lo_c is not None and met_hi_c is not None:
            r_met_lo = float(met_lo_c["sensor_sim"]) - float(sensor_target)
            r_met_hi = float(met_hi_c["sensor_sim"]) - float(sensor_target)
            if r_met_lo * r_met_hi < 0.0:
                try:
                    met_root = float(
                        brentq(
                            lambda m: (evaluate_met_fast(m) or {}).get("sensor_sim", float("nan")) - float(sensor_target),
                            float(met_lo),
                            float(expanded_met_hi),
                            xtol=1e-3,
                            rtol=1e-3,
                            maxiter=40,
                        )
                    )
                    c = evaluate_met_fast(met_root)
                    if c is not None and abs(float(c["sensor_error"])) < best_abs_error:
                        best_candidate = c
                        best_abs_error = abs(float(c["sensor_error"]))
                        best_metabolic = float(c["metabolic_rate_rel"])
                        best_perfusion = float(c["perfusion_factor"])
                except Exception:
                    pass

        # --- Stage 2: brentq over perfusion (fast, no axial; only if stage 1 left error > 0.5) ---
        if best_abs_error > 0.5:
            perf_lo_c = evaluate_perf_fast(float(perf_lo))
            perf_hi_c = evaluate_perf_fast(float(perf_hi))
            for c in (perf_lo_c, perf_hi_c):
                if c is not None and abs(float(c["sensor_error"])) < best_abs_error:
                    best_candidate = c
                    best_abs_error = abs(float(c["sensor_error"]))
                    best_perfusion = float(c["perfusion_factor"])

            if perf_lo_c is not None and perf_hi_c is not None:
                r_perf_lo = float(perf_lo_c["sensor_sim"]) - float(sensor_target)
                r_perf_hi = float(perf_hi_c["sensor_sim"]) - float(sensor_target)
                if r_perf_lo * r_perf_hi < 0.0:
                    try:
                        perf_root = float(
                            brentq(
                                lambda p: (evaluate_perf_fast(p) or {}).get("sensor_sim", float("nan")) - float(sensor_target),
                                float(perf_lo),
                                float(perf_hi),
                                xtol=1e-4,
                                rtol=1e-4,
                                maxiter=60,
                            )
                        )
                        c = evaluate_perf_fast(perf_root)
                        if c is not None and abs(float(c["sensor_error"])) < best_abs_error:
                            best_candidate = c
                            best_abs_error = abs(float(c["sensor_error"]))
                            best_perfusion = float(c["perfusion_factor"])
                    except Exception:
                        pass

        # --- Final: one full include_axial evaluation at the best parameters found ---
        if bool(include_axial):
            final = _eval(best_perfusion, best_metabolic, use_axial=True, met_hi_override=expanded_met_hi)
            if final is not None:
                return final

        return best_candidate

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
        get_numeric_settings: Callable[[], dict[str, float | int]] | None = None,
        temporary_numeric_settings: Callable[[dict[str, float | int]], Any] | None = None,
        build_fast_numeric_settings: Callable[[dict[str, float | int]], dict[str, float | int]] | None = None,
    ) -> None:
        self.solve_axial_capillary_po2 = solve_axial_capillary_po2
        self.effective_p50 = effective_p50
        self.radial_weights = radial_weights
        self.r_vec = r_vec
        self.z_eval = z_eval
        self.R_cap = R_cap
        self.R_tis = R_tis
        self.L_cap = L_cap
        self.get_numeric_settings = get_numeric_settings
        self.temporary_numeric_settings = temporary_numeric_settings
        self.build_fast_numeric_settings = build_fast_numeric_settings

    def _optimize_candidate_locally(
        self,
        *,
        seed_candidate: dict[str, Any],
        P_inlet: float,
        p50_eff: float,
        include_axial: bool,
        sensor_target: float,
        P_v_target: float,
        metabolic_target: float,
        venous_weight: float,
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
        fit_metabolic: bool,
        solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None],
        use_fast_mode: bool = False,
    ) -> dict[str, Any] | None:
        lower_bounds = np.array(
            [
                np.log(max(float(perf_lo), 1e-6)),
                np.log(max(float(p_half_lo), 1e-6)),
                np.log(max(float(met_lo if fit_metabolic else metabolic_target), 1e-6)),
            ],
            dtype=float,
        )
        upper_bounds = np.array(
            [
                np.log(max(float(perf_hi), 1e-6)),
                np.log(max(float(p_half_hi), 1e-6)),
                np.log(max(float(met_hi if fit_metabolic else metabolic_target), 1e-6)),
            ],
            dtype=float,
        )
        initial = np.array(
            [
                np.log(max(float(seed_candidate["perfusion_factor"]), 1e-6)),
                np.log(max(float(seed_candidate["P_half_fit"]), 1e-6)),
                np.log(max(float(seed_candidate.get("metabolic_rate_rel", metabolic_target)), 1e-6)),
            ],
            dtype=float,
        )
        if not fit_metabolic:
            initial[2] = np.log(max(float(metabolic_target), 1e-6))
            lower_bounds[2] = initial[2]
            upper_bounds[2] = initial[2]

        def evaluate_from_log_values(log_values: np.ndarray) -> dict[str, Any] | None:
            bounded = np.minimum(np.maximum(np.asarray(log_values, dtype=float), lower_bounds), upper_bounds)
            values = np.exp(bounded)
            return self._evaluate_single_candidate(
                perfusion_factor=float(values[0]),
                p_half=float(values[1]),
                metabolic_rate_rel=float(values[2]),
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                include_axial=bool(include_axial),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
                solver_cache=solver_cache,
            )

        def objective(log_values: np.ndarray) -> float:
            candidate = evaluate_from_log_values(log_values)
            if candidate is None:
                return float("inf")
            return float(candidate["objective"])

        with self._temporary_numeric_context(use_fast_mode=use_fast_mode):
            best_candidate = evaluate_from_log_values(initial)
            best_log_values = initial
            best_objective = float(best_candidate["objective"]) if best_candidate is not None else float("inf")

            try:
                result = minimize(
                    objective,
                    x0=initial,
                    method="Powell",
                    bounds=[tuple(bound) for bound in np.column_stack((lower_bounds, upper_bounds))],
                    options={
                        "maxiter": 7,
                        "maxfev": 12,
                        "xtol": 8e-2,
                        "ftol": 3e-2,
                    },
                )
                if np.isfinite(float(result.fun)) and float(result.fun) < best_objective:
                    best_log_values = np.asarray(result.x, dtype=float)
                    refined = evaluate_from_log_values(best_log_values)
                    if refined is not None:
                        best_candidate = refined
            except Exception:
                pass

        return best_candidate

    def _local_probe_points(
        self,
        *,
        best_candidate: dict[str, Any],
        perf_lo: float,
        perf_hi: float,
        p_half_lo: float,
        p_half_hi: float,
        met_lo: float,
        met_hi: float,
        fit_metabolic: bool,
    ) -> list[tuple[float, float, float]]:
        perf = float(best_candidate["perfusion_factor"])
        p_half = float(best_candidate["P_half_fit"])
        metabolic_rate_rel = float(best_candidate.get("metabolic_rate_rel", 1.0))

        probe_points = {
            (perf, p_half, metabolic_rate_rel),
            (float(np.clip(perf * 0.88, perf_lo, perf_hi)), p_half, metabolic_rate_rel),
            (float(np.clip(perf * 1.12, perf_lo, perf_hi)), p_half, metabolic_rate_rel),
            (perf, float(np.clip(p_half * 0.85, p_half_lo, p_half_hi)), metabolic_rate_rel),
            (perf, float(np.clip(p_half * 1.18, p_half_lo, p_half_hi)), metabolic_rate_rel),
        }
        if fit_metabolic:
            probe_points.add((perf, p_half, float(np.clip(metabolic_rate_rel * 1.08, met_lo, met_hi))))

        return sorted(probe_points)

    def _bootstrap_uncertainty(
        self,
        candidates: list[dict[str, Any]],
        *,
        sensor_target: float,
        P_v_target: float,
        venous_weight: float,
        bootstrap_samples: int,
    ) -> dict[str, Any]:
        if not candidates or int(bootstrap_samples) <= 0:
            return {
                "bootstrap_samples": int(max(0, bootstrap_samples)),
                "bootstrap_successes": 0,
                "p_half_p10": np.nan,
                "p_half_p90": np.nan,
                "perfusion_p10": np.nan,
                "perfusion_p90": np.nan,
                "metabolic_p10": np.nan,
                "metabolic_p90": np.nan,
                "bootstrap_wide_interval": True,
                "bootstrap_summary": "bootstrap interval unavailable",
            }

        ranked = sorted(candidates, key=lambda item: float(item["objective"]))
        seed = int(
            round(abs(float(sensor_target)) * 100.0)
            + round(abs(float(P_v_target)) * 10.0)
            + 17 * len(ranked)
        ) % (2**32 - 1)
        rng = np.random.default_rng(seed)

        top_sensor_errors = [max(float(item.get("sensor_error", 0.0)), 0.25) for item in ranked[:5]]
        top_venous_errors = [max(float(item.get("venous_error", 0.0)), 0.25) for item in ranked[:5]]
        sensor_sigma = max(0.75, 0.03 * abs(float(sensor_target)), float(np.median(top_sensor_errors)))
        venous_sigma = max(0.50, 0.03 * abs(float(P_v_target)), float(np.median(top_venous_errors)))

        p_half_samples: list[float] = []
        perfusion_samples: list[float] = []
        metabolic_samples: list[float] = []

        for _ in range(int(bootstrap_samples)):
            sampled_sensor_target = max(0.0, float(sensor_target) + float(rng.normal(0.0, sensor_sigma)))
            sampled_venous_target = max(0.0, float(P_v_target) + float(rng.normal(0.0, venous_sigma)))
            best_candidate = min(
                ranked,
                key=lambda item: self._objective_value(
                    sensor_sim=float(item["sensor_sim"]),
                    sensor_target=sampled_sensor_target,
                    P_v_sim=float(item["P_v_sim"]),
                    P_v_target=sampled_venous_target,
                    perfusion_factor=float(item["perfusion_factor"]),
                    p_half=float(item["P_half_fit"]),
                    metabolic_rate_rel=float(item.get("metabolic_rate_rel", 1.0)),
                    metabolic_target=float(item.get("metabolic_target", 1.0)),
                    venous_weight=float(venous_weight),
                ),
            )
            p_half_samples.append(float(best_candidate["P_half_fit"]))
            perfusion_samples.append(float(best_candidate["perfusion_factor"]))
            metabolic_samples.append(float(best_candidate.get("metabolic_rate_rel", 1.0)))

        p_half_arr = np.asarray(p_half_samples, dtype=float)
        perf_arr = np.asarray(perfusion_samples, dtype=float)
        metabolic_arr = np.asarray(metabolic_samples, dtype=float)
        p_half_p10 = float(np.percentile(p_half_arr, 10.0))
        p_half_p90 = float(np.percentile(p_half_arr, 90.0))
        perfusion_p10 = float(np.percentile(perf_arr, 10.0))
        perfusion_p90 = float(np.percentile(perf_arr, 90.0))
        metabolic_p10 = float(np.percentile(metabolic_arr, 10.0))
        metabolic_p90 = float(np.percentile(metabolic_arr, 90.0))
        bootstrap_wide_interval = bool(
            (p_half_p90 / max(p_half_p10, 1e-6) > 4.0)
            or (perfusion_p90 / max(perfusion_p10, 1e-6) > 2.5)
            or (metabolic_p90 / max(metabolic_p10, 1e-6) > 2.0)
        )

        return {
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_successes": int(len(p_half_samples)),
            "p_half_p10": p_half_p10,
            "p_half_p90": p_half_p90,
            "perfusion_p10": perfusion_p10,
            "perfusion_p90": perfusion_p90,
            "metabolic_p10": metabolic_p10,
            "metabolic_p90": metabolic_p90,
            "bootstrap_wide_interval": bootstrap_wide_interval,
            "bootstrap_summary": (
                f"bootstrap-guided band: mitoP50 ≈ {p_half_p10:.3g}–{p_half_p90:.3g} mmHg | "
                f"perfusion ≈ {perfusion_p10:.3g}–{perfusion_p90:.3g}x | "
                f"metabolic rate ≈ {metabolic_p10:.3g}–{metabolic_p90:.3g}x"
            ),
        }

    def _estimate_identifiability(
        self,
        candidates: list[dict[str, Any]],
        *,
        threshold: float,
    ) -> dict[str, Any]:
        """Estimate practical local identifiability from profile-like grid slices."""
        if not candidates:
            return {
                "identifiability": "weak",
                "identifiability_summary": "identifiability unavailable",
                "parameter_correlation": np.nan,
                "profile_p_half_low": np.nan,
                "profile_p_half_high": np.nan,
                "profile_perfusion_low": np.nan,
                "profile_perfusion_high": np.nan,
                "sensitivity_matrix": [[np.nan, np.nan], [np.nan, np.nan]],
                "fisher_determinant": 0.0,
                "fisher_condition": float("inf"),
                "sensitivity_rank": 0,
            }

        p_half_profile: dict[float, float] = {}
        perf_profile: dict[float, float] = {}
        for item in candidates:
            p_half = float(item["P_half_fit"])
            perf = float(item["perfusion_factor"])
            objective = float(item["objective"])
            p_half_profile[p_half] = min(objective, p_half_profile.get(p_half, float("inf")))
            perf_profile[perf] = min(objective, perf_profile.get(perf, float("inf")))

        valid_p_half = sorted(value for value, obj in p_half_profile.items() if obj <= float(threshold))
        valid_perf = sorted(value for value, obj in perf_profile.items() if obj <= float(threshold))

        if not valid_p_half:
            valid_p_half = sorted(p_half_profile.keys())[:1]
        if not valid_perf:
            valid_perf = sorted(perf_profile.keys())[:1]

        p_half_values = np.array([float(item["P_half_fit"]) for item in candidates], dtype=float)
        perf_values = np.array([float(item["perfusion_factor"]) for item in candidates], dtype=float)
        if len(candidates) >= 2 and np.std(np.log(np.maximum(p_half_values, 1e-8))) > 0 and np.std(np.log(np.maximum(perf_values, 1e-8))) > 0:
            parameter_correlation = float(
                np.corrcoef(
                    np.log(np.maximum(p_half_values, 1e-8)),
                    np.log(np.maximum(perf_values, 1e-8)),
                )[0, 1]
            )
        else:
            parameter_correlation = 0.0

        best_item = min(candidates, key=lambda item: float(item["objective"]))
        x_ref = np.array(
            [
                np.log(max(float(best_item["P_half_fit"]), 1e-8)),
                np.log(max(float(best_item["perfusion_factor"]), 1e-8)),
            ],
            dtype=float,
        )
        design_rows: list[list[float]] = []
        sensor_values: list[float] = []
        venous_values: list[float] = []
        for item in candidates:
            x_vals = np.array(
                [
                    np.log(max(float(item["P_half_fit"]), 1e-8)),
                    np.log(max(float(item["perfusion_factor"]), 1e-8)),
                ],
                dtype=float,
            )
            delta = x_vals - x_ref
            design_rows.append([1.0, float(delta[0]), float(delta[1])])
            sensor_values.append(float(item.get("sensor_sim", 0.0)))
            venous_values.append(float(item.get("P_v_sim", 0.0)))

        sensitivity_matrix = np.full((2, 2), np.nan, dtype=float)
        fisher_determinant = 0.0
        fisher_condition = float("inf")
        sensitivity_rank = 0
        design = np.asarray(design_rows, dtype=float)
        if design.shape[0] >= 3 and np.linalg.matrix_rank(design[:, 1:]) >= 2:
            sensor_coef, *_ = np.linalg.lstsq(design, np.asarray(sensor_values, dtype=float), rcond=None)
            venous_coef, *_ = np.linalg.lstsq(design, np.asarray(venous_values, dtype=float), rcond=None)
            sensitivity_matrix = np.array(
                [
                    [float(sensor_coef[1]), float(sensor_coef[2])],
                    [float(venous_coef[1]), float(venous_coef[2])],
                ],
                dtype=float,
            )
            sensitivity_rank = int(np.linalg.matrix_rank(sensitivity_matrix))
            fisher_matrix = sensitivity_matrix.T @ sensitivity_matrix
            fisher_determinant = float(max(np.linalg.det(fisher_matrix), 0.0))
            fisher_condition = float(np.linalg.cond(fisher_matrix)) if sensitivity_rank >= 2 else float("inf")

        profile_p_half_low = float(valid_p_half[0])
        profile_p_half_high = float(valid_p_half[-1])
        profile_perfusion_low = float(valid_perf[0])
        profile_perfusion_high = float(valid_perf[-1])

        p_half_ratio = profile_p_half_high / max(profile_p_half_low, 1e-6)
        perf_ratio = profile_perfusion_high / max(profile_perfusion_low, 1e-6)
        corr_abs = abs(parameter_correlation)
        fisher_ok = bool(np.isfinite(fisher_condition)) and fisher_condition < 250.0 and float(fisher_determinant) > 1e-6

        if p_half_ratio <= 2.0 and perf_ratio <= 1.6 and corr_abs < 0.6 and fisher_ok and sensitivity_rank >= 2:
            identifiability = "strong"
        elif p_half_ratio <= 4.0 and perf_ratio <= 2.5 and corr_abs < 0.85 and sensitivity_rank >= 2:
            identifiability = "moderate"
        else:
            identifiability = "weak"

        fisher_phrase = (
            f"local Fisher condition {fisher_condition:.2g}"
            if np.isfinite(fisher_condition)
            else "local Fisher condition unavailable"
        )
        identifiability_summary = (
            f"identifiability appears {identifiability}: profile-supported mitoP50 range ≈ "
            f"{profile_p_half_low:.3g}–{profile_p_half_high:.3g} mmHg, perfusion ≈ "
            f"{profile_perfusion_low:.3g}–{profile_perfusion_high:.3g}x, "
            f"coupling correlation {parameter_correlation:+.2f}, {fisher_phrase}"
        )

        return {
            "identifiability": identifiability,
            "identifiability_summary": identifiability_summary,
            "parameter_correlation": parameter_correlation,
            "profile_p_half_low": profile_p_half_low,
            "profile_p_half_high": profile_p_half_high,
            "profile_perfusion_low": profile_perfusion_low,
            "profile_perfusion_high": profile_perfusion_high,
            "sensitivity_matrix": sensitivity_matrix.tolist(),
            "fisher_determinant": fisher_determinant,
            "fisher_condition": fisher_condition,
            "sensitivity_rank": sensitivity_rank,
        }

    def _estimate_uncertainty(
        self,
        candidates: list[dict[str, Any]],
        best_objective: float,
        *,
        sensor_target: float,
        P_v_target: float,
        venous_weight: float,
        bootstrap_samples: int = 80,
    ) -> dict[str, Any]:
        """Summarize a practical near-optimal and bootstrap-guided parameter band.

        This remains intentionally lightweight: it is not a formal Bayesian posterior,
        but it gives the user a clearer view of the locally plausible parameter region.
        """
        if not candidates:
            return {
                "method": "practical near-optimal band with bootstrap unavailable",
                "candidate_count": 0,
                "p_half_low": np.nan,
                "p_half_high": np.nan,
                "perfusion_low": np.nan,
                "perfusion_high": np.nan,
                "metabolic_low": np.nan,
                "metabolic_high": np.nan,
                "objective_threshold": np.nan,
                "bootstrap_samples": int(max(0, bootstrap_samples)),
                "bootstrap_successes": 0,
                "p_half_p10": np.nan,
                "p_half_p90": np.nan,
                "perfusion_p10": np.nan,
                "perfusion_p90": np.nan,
                "metabolic_p10": np.nan,
                "metabolic_p90": np.nan,
                "identifiability": "weak",
                "identifiability_summary": "identifiability unavailable",
                "parameter_correlation": np.nan,
                "profile_p_half_low": np.nan,
                "profile_p_half_high": np.nan,
                "profile_perfusion_low": np.nan,
                "profile_perfusion_high": np.nan,
                "sensitivity_matrix": [[np.nan, np.nan], [np.nan, np.nan]],
                "fisher_determinant": np.nan,
                "fisher_condition": np.nan,
                "sensitivity_rank": 0,
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
        metabolic_values = np.array([float(item.get("metabolic_rate_rel", 1.0)) for item in near_optimal], dtype=float)

        p_half_low = float(np.min(p_half_values))
        p_half_high = float(np.max(p_half_values))
        perfusion_low = float(np.min(perf_values))
        perfusion_high = float(np.max(perf_values))
        metabolic_low = float(np.min(metabolic_values))
        metabolic_high = float(np.max(metabolic_values))

        bootstrap = self._bootstrap_uncertainty(
            ranked,
            sensor_target=float(sensor_target),
            P_v_target=float(P_v_target),
            venous_weight=float(venous_weight),
            bootstrap_samples=int(bootstrap_samples),
        )
        identifiability = self._estimate_identifiability(near_optimal, threshold=float(threshold))
        wide_interval = bool(
            (p_half_high / max(p_half_low, 1e-6) > 4.0)
            or (perfusion_high / max(perfusion_low, 1e-6) > 2.5)
            or (metabolic_high / max(metabolic_low, 1e-6) > 2.0)
            or bool(bootstrap.get("bootstrap_wide_interval", False))
            or str(identifiability.get("identifiability", "moderate")) == "weak"
        )

        return {
            "method": "practical near-optimal grid band with measurement-perturbation bootstrap",
            "candidate_count": int(len(near_optimal)),
            "p_half_low": p_half_low,
            "p_half_high": p_half_high,
            "perfusion_low": perfusion_low,
            "perfusion_high": perfusion_high,
            "metabolic_low": metabolic_low,
            "metabolic_high": metabolic_high,
            "objective_threshold": float(threshold),
            "bootstrap_samples": int(bootstrap.get("bootstrap_samples", 0)),
            "bootstrap_successes": int(bootstrap.get("bootstrap_successes", 0)),
            "p_half_p10": float(bootstrap.get("p_half_p10", np.nan)),
            "p_half_p90": float(bootstrap.get("p_half_p90", np.nan)),
            "perfusion_p10": float(bootstrap.get("perfusion_p10", np.nan)),
            "perfusion_p90": float(bootstrap.get("perfusion_p90", np.nan)),
            "metabolic_p10": float(bootstrap.get("metabolic_p10", np.nan)),
            "metabolic_p90": float(bootstrap.get("metabolic_p90", np.nan)),
            "identifiability": str(identifiability.get("identifiability", "weak")),
            "identifiability_summary": str(identifiability.get("identifiability_summary", "identifiability unavailable")),
            "parameter_correlation": float(identifiability.get("parameter_correlation", np.nan)),
            "profile_p_half_low": float(identifiability.get("profile_p_half_low", np.nan)),
            "profile_p_half_high": float(identifiability.get("profile_p_half_high", np.nan)),
            "profile_perfusion_low": float(identifiability.get("profile_perfusion_low", np.nan)),
            "profile_perfusion_high": float(identifiability.get("profile_perfusion_high", np.nan)),
            "sensitivity_matrix": identifiability.get("sensitivity_matrix", [[np.nan, np.nan], [np.nan, np.nan]]),
            "fisher_determinant": float(identifiability.get("fisher_determinant", np.nan)),
            "fisher_condition": float(identifiability.get("fisher_condition", np.nan)),
            "sensitivity_rank": int(identifiability.get("sensitivity_rank", 0)),
            "wide_interval": wide_interval,
            "summary": (
                f"bootstrap-guided mitoP50 ≈ {float(bootstrap.get('p_half_p10', p_half_low)):.3g}–"
                f"{float(bootstrap.get('p_half_p90', p_half_high)):.3g} mmHg | "
                f"perfusion ≈ {float(bootstrap.get('perfusion_p10', perfusion_low)):.3g}–"
                f"{float(bootstrap.get('perfusion_p90', perfusion_high)):.3g}x | "
                f"metabolic rate ≈ {float(bootstrap.get('metabolic_p10', metabolic_low)):.3g}–"
                f"{float(bootstrap.get('metabolic_p90', metabolic_high)):.3g}x | "
                f"identifiability: {str(identifiability.get('identifiability', 'weak'))} "
                f"(corr {float(identifiability.get('parameter_correlation', np.nan)):+.2f}) "
                f"({int(bootstrap.get('bootstrap_successes', 0))}/{int(bootstrap.get('bootstrap_samples', 0))} bootstrap refits; "
                f"{len(near_optimal)} near-optimal grid fits)"
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
        below_1 = float(scenario.get("fraction_below_1", 0.0))
        below_5 = float(scenario.get("fraction_below_5", 0.0))
        below_10 = float(scenario.get("fraction_below_10", 0.0))
        below_15 = float(scenario.get("fraction_below_15", 0.0))

        if below_1 > 0.01 or below_5 >= 0.10 or below_10 >= 0.35:
            return "red", "serious risk of severe hypoxia"
        if below_5 >= 0.03 or below_10 >= 0.20:
            return "orange", "high probability of major hypoxic spots"
        if below_10 >= 0.05 or below_15 >= 0.30:
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
        metabolic_rate_rel: float,
        metabolic_target: float,
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
            f"the best fit required {perfusion_phrase}, mitoP50 near {p_half_fit:.2f} mmHg, "
            f"and a relative tissue metabolic O2 demand near {metabolic_rate_rel:.2f}x (input target {metabolic_target:.2f}x). "
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
            P_c_axial, _, _ = self._call_solver(
                P_inlet=P_inlet,
                P_half=P_half_candidate,
                p50_eff=p50_eff,
                include_axial=include_axial,
                perfusion_factor=perfusion_factor,
                metabolic_rate_rel=1.0,
            )
            return float(P_c_axial[-1])

        def _residual(P_half_candidate: float) -> float:
            return _simulate_venous(P_half_candidate) - P_v_target

        lo, hi = 0.1, 5.0
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
        metabolic_target: float = 1.0,
        fit_metabolic: bool = True,
        perfusion_bounds: tuple[float, float] = (0.35, 5.0),
        p_half_bounds: tuple[float, float] = (0.1, 5.0),
        metabolic_bounds: tuple[float, float] = (0.4, 2.5),
        include_axial: bool = True,
        venous_weight: float = 0.15,
        bootstrap_samples: int = 80,
        search_strategy: str = "optimized",
    ) -> dict[str, Any]:
        p50_eff = self.effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)
        perf_lo, perf_hi = float(perfusion_bounds[0]), float(perfusion_bounds[1])
        p_half_lo, p_half_hi = float(p_half_bounds[0]), float(p_half_bounds[1])
        met_lo, met_hi = float(metabolic_bounds[0]), float(metabolic_bounds[1])
        metabolic_target = float(np.clip(float(metabolic_target), met_lo, met_hi))
        search_mode = str(search_strategy or "optimized").strip().lower()
        if search_mode not in {"optimized", "legacy_grid"}:
            raise ValueError(f"Unsupported search strategy: {search_strategy}")

        perf_candidates = np.unique(np.concatenate(([1.0], np.geomspace(perf_lo, perf_hi, 15))))
        p_half_candidates = np.unique(np.concatenate(([1.0], np.geomspace(p_half_lo, p_half_hi, 22))))
        if bool(fit_metabolic):
            metabolic_candidates = np.unique(
                np.concatenate(([1.0, metabolic_target], np.geomspace(met_lo, met_hi, 11)))
            )
        else:
            metabolic_candidates = np.array([metabolic_target], dtype=float)
        exhaustive_value_points = [
            (float(perf), float(p_half), float(metabolic_rate_rel))
            for perf in perf_candidates
            for p_half in p_half_candidates
            for metabolic_rate_rel in metabolic_candidates
        ]

        if search_mode == "legacy_grid":
            evaluated_candidates = self._evaluate_candidate_points(
                exhaustive_value_points,
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                include_axial=bool(include_axial),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
                parallel=True,
            )
            best = min(evaluated_candidates, key=lambda item: float(item["objective"])) if evaluated_candidates else None
        else:
            full_index_points = self._screen_candidate_points(
                perf_candidates=perf_candidates,
                p_half_candidates=p_half_candidates,
                metabolic_candidates=metabolic_candidates,
                P_inlet=float(P_inlet),
                p50_eff=float(p50_eff),
                sensor_target=float(sensor_target),
                P_v_target=float(P_v_target),
                metabolic_target=float(metabolic_target),
                venous_weight=float(venous_weight),
                perf_lo=float(perf_lo),
                perf_hi=float(perf_hi),
                p_half_lo=float(p_half_lo),
                p_half_hi=float(p_half_hi),
                met_lo=float(met_lo),
                met_hi=float(met_hi),
            )

            if full_index_points:
                candidate_value_points = [
                    (
                        float(perf_candidates[perf_index]),
                        float(p_half_candidates[p_half_index]),
                        float(metabolic_candidates[metabolic_index]),
                    )
                    for perf_index, p_half_index, metabolic_index in full_index_points
                ]
            else:
                candidate_value_points = exhaustive_value_points

            screening_candidates: list[dict[str, Any]] = []
            with self._temporary_numeric_context(use_fast_mode=True):
                screening_candidates = self._evaluate_candidate_points(
                    candidate_value_points,
                    P_inlet=float(P_inlet),
                    p50_eff=float(p50_eff),
                    include_axial=False,
                    sensor_target=float(sensor_target),
                    P_v_target=float(P_v_target),
                    metabolic_target=float(metabolic_target),
                    venous_weight=float(venous_weight),
                    perf_lo=float(perf_lo),
                    perf_hi=float(perf_hi),
                    p_half_lo=float(p_half_lo),
                    p_half_hi=float(p_half_hi),
                    met_lo=float(met_lo),
                    met_hi=float(met_hi),
                )

            full_solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None] = {}
            evaluated_candidates = []

            if screening_candidates:
                local_seeds = sorted(screening_candidates, key=lambda item: float(item["objective"]))[:1]
                optimized_candidates = [
                    self._optimize_candidate_locally(
                        seed_candidate=seed,
                        P_inlet=float(P_inlet),
                        p50_eff=float(p50_eff),
                        include_axial=bool(include_axial),
                        sensor_target=float(sensor_target),
                        P_v_target=float(P_v_target),
                        metabolic_target=float(metabolic_target),
                        venous_weight=float(venous_weight),
                        perf_lo=float(perf_lo),
                        perf_hi=float(perf_hi),
                        p_half_lo=float(p_half_lo),
                        p_half_hi=float(p_half_hi),
                        met_lo=float(met_lo),
                        met_hi=float(met_hi),
                        fit_metabolic=bool(fit_metabolic),
                        solver_cache=full_solver_cache,
                        use_fast_mode=True,
                    )
                    for seed in local_seeds
                ]
                evaluated_candidates = [candidate for candidate in optimized_candidates if candidate is not None]

            best = min(evaluated_candidates, key=lambda item: float(item["objective"])) if evaluated_candidates else None

            if best is not None:
                probe_points = self._local_probe_points(
                    best_candidate=best,
                    perf_lo=float(perf_lo),
                    perf_hi=float(perf_hi),
                    p_half_lo=float(p_half_lo),
                    p_half_hi=float(p_half_hi),
                    met_lo=float(met_lo),
                    met_hi=float(met_hi),
                    fit_metabolic=bool(fit_metabolic),
                )
                evaluated_candidates = self._evaluate_candidate_points(
                    probe_points,
                    P_inlet=float(P_inlet),
                    p50_eff=float(p50_eff),
                    include_axial=bool(include_axial),
                    sensor_target=float(sensor_target),
                    P_v_target=float(P_v_target),
                    metabolic_target=float(metabolic_target),
                    venous_weight=float(venous_weight),
                    perf_lo=float(perf_lo),
                    perf_hi=float(perf_hi),
                    p_half_lo=float(p_half_lo),
                    p_half_hi=float(p_half_hi),
                    met_lo=float(met_lo),
                    met_hi=float(met_hi),
                    solver_cache=full_solver_cache,
                    parallel=True,
                )
                best = min(evaluated_candidates, key=lambda item: float(item["objective"])) if evaluated_candidates else best

            if best is None:
                evaluated_candidates = self._evaluate_candidate_points(
                    exhaustive_value_points,
                    P_inlet=float(P_inlet),
                    p50_eff=float(p50_eff),
                    include_axial=bool(include_axial),
                    sensor_target=float(sensor_target),
                    P_v_target=float(P_v_target),
                    metabolic_target=float(metabolic_target),
                    venous_weight=float(venous_weight),
                    perf_lo=float(perf_lo),
                    perf_hi=float(perf_hi),
                    p_half_lo=float(p_half_lo),
                    p_half_hi=float(p_half_hi),
                    met_lo=float(met_lo),
                    met_hi=float(met_hi),
                    parallel=True,
                )
                best = min(evaluated_candidates, key=lambda item: float(item["objective"])) if evaluated_candidates else None

        if best is None:
            raise RuntimeError("Could not fit Krogh parameters for the selected diagnostic inputs.")

        full_solver_cache: dict[tuple[float, float, float, bool], dict[str, Any] | None] = {}
        best = self._refine_candidate_to_sensor_target(
            seed_candidate=best,
            P_inlet=float(P_inlet),
            p50_eff=float(p50_eff),
            include_axial=bool(include_axial),
            sensor_target=float(sensor_target),
            P_v_target=float(P_v_target),
            metabolic_target=float(metabolic_target),
            venous_weight=float(venous_weight),
            perf_lo=float(perf_lo),
            perf_hi=float(perf_hi),
            p_half_lo=float(p_half_lo),
            p_half_hi=float(p_half_hi),
            met_lo=float(met_lo),
            met_hi=float(met_hi),
            solver_cache=full_solver_cache,
        )
        if best is not None:
            evaluated_candidates = list(evaluated_candidates)
            evaluated_candidates.append(best)

        best["search_strategy"] = search_mode
        best["uncertainty"] = self._estimate_uncertainty(
            evaluated_candidates,
            float(best["objective"]),
            sensor_target=float(sensor_target),
            P_v_target=float(P_v_target),
            venous_weight=float(venous_weight),
            bootstrap_samples=int(bootstrap_samples),
        )
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
        metabolic_rate_rel = float(fit.get("metabolic_rate_rel", 1.0))
        metabolic_target = float(fit.get("metabolic_target", 1.0))
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
            metabolic_rate_rel=metabolic_rate_rel,
            metabolic_target=metabolic_target,
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
            "metabolic_rate_rel": metabolic_rate_rel,
            "metabolic_target": metabolic_target,
            "sensor_target": float(sensor_po2),
            "sensor_error": float(fit["sensor_error"]),
            "venous_error": float(fit["venous_error"]),
            "fit_warning": bool(fit["fit_warning"]),
            "fit_boundary_hit": bool(fit.get("fit_boundary_hit", False)),
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
            "tissue_po2": tissue_po2,
            "po2_min_plot": 0.0,
            "po2_max_plot": max(float(po2) * 1.05, 30.0),
            "p_venous": float(P_c_axial[-1]),
            "p_tis_min": float(np.min(tissue_po2)),
            "sensor_avg": float(P_avg.mean()),
            "sensor_sim": float(fit["sensor_sim"]),
        }


__all__ = ["KroghReconstructor"]
