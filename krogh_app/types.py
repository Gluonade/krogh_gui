"""Shared dataclasses for the Krogh GUI project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericSettings:
    ode_rtol: float
    ode_atol: float
    ode_max_step: float
    axial_diffusion_max_iter: int
    axial_diffusion_tol: float
    axial_coupling_max_iter: int
    axial_coupling_tol: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SingleCaseInput:
    P_inlet: float
    P_half: float
    pH: float
    pCO2: float
    temp_c: float
    perfusion_factor: float
    include_axial_diffusion: bool = True
    high_po2_threshold_primary: float = 100.0
    high_po2_threshold_secondary: float = 200.0
    additional_high_po2_thresholds: tuple[float, ...] = ()
    relative_high_po2_thresholds_percent: tuple[float, ...] = (90.0, 50.0, 30.0)
    relative_high_po2_reference: str = "inlet"


@dataclass(frozen=True)
class DiagnosticRunInput:
    po2: float
    pco2: float
    pH: float
    temperature_c: float
    sensor_po2: float
    hemoglobin_g_dl: float = 13.5
    venous_sat_percent: float = 75.0
    yellow_threshold: float = 0.25
    orange_threshold: float = 0.50
    red_threshold: float = 0.75


@dataclass
class SimulationResult:
    summary: dict[str, Any] = field(default_factory=dict)
    capillary_profile: Any = None
    tissue_field: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "capillary_profile": self.capillary_profile,
            "tissue_field": self.tissue_field,
            "metadata": self.metadata,
        }
