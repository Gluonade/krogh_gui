"""Diagnostic service layer for the Krogh GUI project.

This module provides a small object-oriented wrapper around the existing
probabilistic oxygenation diagnostic logic so the GUI can delegate work to a
focused service instead of calling the MVP module directly.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import DiagnosticRunInput

try:
    from oxygenation_diagnostic_mvp import OxygenationInput, alert_decision, effective_p50
except Exception:
    OxygenationInput = None
    alert_decision = None
    effective_p50 = None


class DiagnosticEngine:
    """Thin service wrapper for diagnostic evaluation and target derivation."""

    @property
    def is_available(self) -> bool:
        return OxygenationInput is not None and alert_decision is not None and effective_p50 is not None

    def evaluate(self, diagnostic_input: DiagnosticRunInput | dict[str, Any]) -> dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("Diagnostic module oxygenation_diagnostic_mvp.py could not be loaded.")

        if isinstance(diagnostic_input, dict):
            diagnostic_input = DiagnosticRunInput(**diagnostic_input)

        return alert_decision(
            OxygenationInput(
                po2=float(diagnostic_input.po2),
                pco2=float(diagnostic_input.pco2),
                pH=float(diagnostic_input.pH),
                temperature_c=float(diagnostic_input.temperature_c),
                sensor_po2=float(diagnostic_input.sensor_po2),
                hemoglobin_g_dl=float(diagnostic_input.hemoglobin_g_dl),
                venous_sat_percent=float(diagnostic_input.venous_sat_percent),
            ),
            yellow_threshold=float(diagnostic_input.yellow_threshold),
            orange_threshold=float(diagnostic_input.orange_threshold),
            red_threshold=float(diagnostic_input.red_threshold),
        )

    def to_record(self, diagnostic_input: DiagnosticRunInput) -> dict[str, Any]:
        return asdict(diagnostic_input)

    def validate_thresholds(self, yellow_threshold: float, orange_threshold: float, red_threshold: float) -> bool:
        return 0.0 <= yellow_threshold <= orange_threshold <= red_threshold <= 1.0

    def venous_target_po2(self, pH: float, pco2: float, temperature_c: float, venous_sat_percent: float) -> float:
        if not self.is_available:
            raise RuntimeError("Diagnostic module oxygenation_diagnostic_mvp.py could not be loaded.")
        p50_eff = effective_p50(pH=pH, pco2=pco2, temperature_c=temperature_c)
        svo2 = max(0.01, min(0.99, float(venous_sat_percent) / 100.0))
        return float(p50_eff * (svo2 / (1.0 - svo2)) ** (1.0 / 2.7))


__all__ = ["DiagnosticEngine"]
