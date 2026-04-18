"""Core package for the Krogh GUI application.

This package is introduced as a first safe refactoring step.
The existing GUI remains fully usable while constants, shared types,
and future service classes are gradually moved into dedicated modules.
"""

from .localization import TranslationManager
from .types import NumericSettings, SingleCaseInput, DiagnosticRunInput, SimulationResult

__all__ = [
    "TranslationManager",
    "NumericSettings",
    "SingleCaseInput",
    "DiagnosticRunInput",
    "SimulationResult",
]
