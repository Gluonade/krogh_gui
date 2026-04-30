"""Probabilistic oxygenation diagnostic MVP.

This module provides a lightweight, dependency-free diagnostic layer for the
Tkinter Krogh GUI. It combines blood-gas values and a tissue/sensor PO2 reading
into a probabilistic risk estimate and an alert color.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict

from krogh_app.constants import (
    BOHR_COEFF,
    CO2_COEFF,
    N_HILL,
    P50_STD,
    PCO2_REF,
    PH_REF,
    TEMP_COEFF,
    TEMP_REF,
)


@dataclass(frozen=True)
class OxygenationInput:
    po2: float
    pco2: float
    pH: float
    temperature_c: float
    sensor_po2: float
    hemoglobin_g_dl: float = 13.5
    venous_sat_percent: float = 75.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(score_map: Dict[str, float]) -> Dict[str, float]:
    max_score = max(score_map.values())
    exp_map = {key: math.exp(max(-60.0, min(60.0, value - max_score))) for key, value in score_map.items()}
    total = sum(exp_map.values()) or 1.0
    return {key: value / total for key, value in exp_map.items()}


def effective_p50(pH: float, pco2: float, temperature_c: float) -> float:
    pco2_safe = max(float(pco2), 1e-6)
    log_shift = (
        BOHR_COEFF * (float(pH) - PH_REF)
        + CO2_COEFF * math.log10(pco2_safe / PCO2_REF)
        + TEMP_COEFF * (float(temperature_c) - TEMP_REF)
    )
    return P50_STD * (10.0 ** log_shift)


def hill_saturation(po2: float, p50_eff: float | None = None) -> float:
    if p50_eff is None:
        p50_eff = P50_STD
    po2_safe = max(float(po2), 1e-9)
    pn = po2_safe ** N_HILL
    return pn / (pn + float(p50_eff) ** N_HILL)


def _describe_feature_risks(feature_risks: Dict[str, float]) -> tuple[list[str], str]:
    labels = {
        "sensor_risk": "low tissue/sensor PO2",
        "gap_risk": "a widened arterial-to-tissue PO2 gap",
        "po2_risk": "reduced arterial PO2",
        "acid_risk": "acidosis",
        "hypercapnia_risk": "hypercapnia",
        "anemia_risk": "low hemoglobin",
        "venous_risk": "reduced venous saturation",
        "temperature_risk": "temperature-related stress",
    }
    ranked = sorted(feature_risks.items(), key=lambda item: float(item[1]), reverse=True)
    dominant = [labels.get(key, key.replace("_", " ")) for key, value in ranked if float(value) >= 0.30][:3]
    if not dominant:
        dominant = [labels.get(ranked[0][0], ranked[0][0].replace("_", " "))] if ranked else []
    summary = "Main alert-score drivers: " + ", ".join(dominant) + "." if dominant else ""
    return dominant, summary


def _state_probabilities(risk_score: float, compensation_gap: float, sensor_po2: float, po2: float) -> Dict[str, float]:
    # Primary state assignment follows physiologically meaningful PO2 bands,
    # while the composite risk score and compensation gap act as secondary modifiers.
    po2_band_centers = {
        "normoxia": 50.0,
        "intermediate_oxygenation": 30.0,
        "low_oxygenation_approaching_critical": 15.0,
        "hypoxia": 6.0,
        "profound_hypoxia": 1.0,
    }
    po2_band_sigmas = {
        "normoxia": 7.5,
        "intermediate_oxygenation": 6.0,
        "low_oxygenation_approaching_critical": 4.0,
        "hypoxia": 2.6,
        "profound_hypoxia": 1.1,
    }

    scores: Dict[str, float] = {}
    for state, center in po2_band_centers.items():
        sigma = po2_band_sigmas[state]
        scores[state] = -((sensor_po2 - center) ** 2) / (2.0 * sigma ** 2)

    scores["normoxia"] += 0.55 * _clamp((sensor_po2 - 40.0) / 25.0)
    scores["intermediate_oxygenation"] += 0.30 * _clamp((40.0 - sensor_po2) / 20.0)
    scores["low_oxygenation_approaching_critical"] += 0.30 * _clamp((20.0 - sensor_po2) / 12.0)
    scores["hypoxia"] += 0.45 * _clamp((10.0 - sensor_po2) / 8.0)
    scores["profound_hypoxia"] += 0.70 * _clamp((2.0 - sensor_po2) / 2.0)

    scores["intermediate_oxygenation"] += 0.25 * risk_score
    scores["low_oxygenation_approaching_critical"] += 0.40 * risk_score + 0.25 * compensation_gap
    scores["hypoxia"] += 0.55 * risk_score + 0.35 * compensation_gap
    scores["profound_hypoxia"] += 0.80 * risk_score + 0.40 * compensation_gap

    if sensor_po2 < 2.0:
        scores["profound_hypoxia"] += 2.0
    elif sensor_po2 < 10.0:
        scores["hypoxia"] += 1.2
    elif sensor_po2 < 20.0:
        scores["low_oxygenation_approaching_critical"] += 0.8
    elif sensor_po2 < 40.0:
        scores["intermediate_oxygenation"] += 0.55

    if po2 < 30.0:
        scores["hypoxia"] += 0.45
    if po2 < 20.0:
        scores["profound_hypoxia"] += 0.55

    return _softmax(scores)


def alert_decision(
    oxygen_input: OxygenationInput,
    yellow_threshold: float = 0.25,
    orange_threshold: float = 0.50,
    red_threshold: float = 0.75,
) -> Dict[str, Any]:
    if not (0.0 <= yellow_threshold <= orange_threshold <= red_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy 0 <= yellow <= orange <= red <= 1.")

    data = OxygenationInput(
        po2=float(oxygen_input.po2),
        pco2=float(oxygen_input.pco2),
        pH=float(oxygen_input.pH),
        temperature_c=float(oxygen_input.temperature_c),
        sensor_po2=float(oxygen_input.sensor_po2),
        hemoglobin_g_dl=float(oxygen_input.hemoglobin_g_dl),
        venous_sat_percent=float(oxygen_input.venous_sat_percent),
    )

    p50_eff = effective_p50(data.pH, data.pco2, data.temperature_c)
    sa_percent = 100.0 * hill_saturation(data.po2, p50_eff=p50_eff)
    compensation_gap = _clamp((data.po2 - data.sensor_po2) / 80.0)

    po2_risk = _sigmoid((55.0 - data.po2) / 12.0)
    sensor_risk = _sigmoid((35.0 - data.sensor_po2) / 10.0)
    acid_risk = _sigmoid((7.32 - data.pH) / 0.06)
    hypercapnia_risk = _sigmoid((data.pco2 - 48.0) / 8.0)
    anemia_risk = _sigmoid((11.5 - data.hemoglobin_g_dl) / 1.5)
    venous_risk = _sigmoid((68.0 - data.venous_sat_percent) / 7.0)
    temperature_risk = max(
        _sigmoid((35.0 - data.temperature_c) / 0.8),
        _sigmoid((data.temperature_c - 38.5) / 0.8),
    )
    gap_risk = _sigmoid(((data.po2 - data.sensor_po2) - 45.0) / 15.0)

    risk_score = (
        0.22 * po2_risk
        + 0.24 * sensor_risk
        + 0.08 * gap_risk
        + 0.12 * acid_risk
        + 0.10 * hypercapnia_risk
        + 0.08 * anemia_risk
        + 0.06 * venous_risk
        + 0.02 * temperature_risk
    )
    risk_score = _clamp(risk_score)

    if data.sensor_po2 < 15.0 or data.po2 < 25.0:
        risk_score = max(risk_score, 0.90)
    elif data.sensor_po2 < 25.0 or data.po2 < 35.0:
        risk_score = max(risk_score, 0.72)

    probabilities = _state_probabilities(risk_score, compensation_gap, data.sensor_po2, data.po2)
    predicted_state = max(probabilities, key=probabilities.get)

    sorted_probs = sorted(probabilities.values(), reverse=True)
    confidence = float(sorted_probs[0])
    certainty = float(_clamp(confidence - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)))

    if risk_score >= red_threshold:
        alert_level = "red"
    elif risk_score >= orange_threshold:
        alert_level = "orange"
    elif risk_score >= yellow_threshold:
        alert_level = "yellow"
    else:
        alert_level = "green"

    feature_risks = {
        "po2_risk": float(po2_risk),
        "sensor_risk": float(sensor_risk),
        "acid_risk": float(acid_risk),
        "hypercapnia_risk": float(hypercapnia_risk),
        "anemia_risk": float(anemia_risk),
        "venous_risk": float(venous_risk),
        "temperature_risk": float(temperature_risk),
        "gap_risk": float(gap_risk),
    }
    dominant_drivers, driver_summary = _describe_feature_risks(feature_risks)

    return {
        "predicted_state": predicted_state,
        "risk_score": float(risk_score),
        "alert_level": alert_level,
        "confidence": confidence,
        "certainty": certainty,
        "p_normoxia": float(probabilities["normoxia"]),
        "p_intermediate_oxygenation": float(probabilities["intermediate_oxygenation"]),
        "p_low_oxygenation_approaching_critical": float(probabilities["low_oxygenation_approaching_critical"]),
        "p_hypoxia": float(probabilities["hypoxia"]),
        "p_profound_hypoxia": float(probabilities["profound_hypoxia"]),
        # Backward-compatible aliases for existing GUI/report paths.
        "p_mild_hypoxia": float(probabilities["intermediate_oxygenation"]),
        "p_compensated_hypoxia": float(probabilities["low_oxygenation_approaching_critical"]),
        "p_severe_hypoxia": float(probabilities["hypoxia"]),
        "estimated_sa_percent": float(sa_percent),
        "p50_eff": float(p50_eff),
        "feature_risks": feature_risks,
        "dominant_risk_drivers": dominant_drivers,
        "driver_summary": driver_summary,
        "input": asdict(data),
    }


__all__ = ["OxygenationInput", "alert_decision", "effective_p50", "hill_saturation"]
