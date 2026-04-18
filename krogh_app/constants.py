"""Shared physical and numerical constants for the Krogh GUI project.

This module is introduced as a safe first extraction step so that the
scientific defaults live in one place and can later be reused by model,
diagnostics, and UI service classes.
"""

from __future__ import annotations

import numpy as np

# Shared model constants
n_hill = 2.7
P50 = 26.8
C_Hb = 0.20
alpha = 3.0e-5

PH_REF = 7.40
PCO2_REF = 40.0
TEMP_REF = 37.0
BOHR_COEFF = -0.48
CO2_COEFF = 0.06
TEMP_COEFF = 0.024

R_cap = 5e-4
R_tis = 100e-4
M_rate = 5.8e-4
K_diff = 2e-9
L_cap = 0.1
v_blood = 0.47
MIN_CONSUMPTION_FRACTION = 0.05
Q_flow = v_blood * np.pi * R_cap**2

NZ = 80
NR = 80
z_eval = np.linspace(0, L_cap, NZ)
r_vec = np.linspace(R_cap, R_tis, NR)
dr = (R_tis - R_cap) / (NR - 1)
dz = L_cap / (NZ - 1)
radial_weights = r_vec / np.sum(r_vec * dr)

AXIAL_DIFFUSION_MAX_ITER = 140
AXIAL_DIFFUSION_TOL = 1e-5
AXIAL_DIFFUSION_RELAX = 0.45
AXIAL_COUPLING_MAX_ITER = 5
AXIAL_COUPLING_TOL = 1e-4
CAPILLARY_ODE_RTOL = 1e-9
CAPILLARY_ODE_ATOL = 1e-11
CAPILLARY_ODE_MAX_STEP = dz

NUMERIC_SETTINGS_FIELDS = (
    ("ode_rtol", "numeric_ode_rtol", float),
    ("ode_atol", "numeric_ode_atol", float),
    ("ode_max_step", "numeric_ode_max_step", float),
    ("axial_diffusion_max_iter", "numeric_axial_diffusion_max_iter", int),
    ("axial_diffusion_tol", "numeric_axial_diffusion_tol", float),
    ("axial_coupling_max_iter", "numeric_axial_coupling_max_iter", int),
    ("axial_coupling_tol", "numeric_axial_coupling_tol", float),
)

NUMERIC_SETTINGS_TYPES = {
    key: value_type for key, _, value_type in NUMERIC_SETTINGS_FIELDS
}

NUMERIC_SETTING_SPECS = {
    "ode_rtol": {
        "min": 1e-12,
        "max": 1e-6,
        "default": CAPILLARY_ODE_RTOL,
        "description_en": "Relative target accuracy of the adaptive capillary ODE solver. Smaller values reduce hidden solver drift in sensitive parameter sweeps but increase runtime.",
        "description_de": "Relatives Fehlerziel des adaptiven Kapillar-ODE-Solvers. Kleinere Werte reduzieren versteckte Solverdrift in sensiblen Parameter-Sweeps, erhoehen aber die Rechenzeit.",
    },
    "ode_atol": {
        "min": 1e-14,
        "max": 1e-8,
        "default": CAPILLARY_ODE_ATOL,
        "description_en": "Absolute target accuracy of the capillary ODE solver. This matters most when PO2 values approach very low levels near the hypoxic regime.",
        "description_de": "Absolutes Fehlerziel des Kapillar-ODE-Solvers. Dieser Wert ist besonders wichtig, wenn PO2-Werte in den hypoxischen Bereich absinken.",
    },
    "ode_max_step": {
        "min": max(dz / 20.0, 1e-6),
        "max": 0.02,
        "default": CAPILLARY_ODE_MAX_STEP,
        "description_en": "Upper bound for one adaptive ODE step along the capillary axis. Smaller steps improve axial resolution and stabilize sharp gradients, especially at low perfusion.",
        "description_de": "Obere Grenze fuer einen adaptiven ODE-Schritt entlang der Kapillare. Kleinere Schritte verbessern die axiale Aufloesung und stabilisieren steile Gradienten, vor allem bei niedriger Perfusion.",
    },
    "axial_diffusion_max_iter": {
        "min": 20,
        "max": 2000,
        "default": AXIAL_DIFFUSION_MAX_ITER,
        "description_en": "Maximum number of inner iterations for the 2D tissue diffusion solve. Increase it when the tissue field converges too slowly; extremely high values mainly cost time.",
        "description_de": "Maximale Zahl innerer Iterationen fuer die 2D-Gewebediffusionsloesung. Hoehere Werte helfen nur, wenn das Gewebefeld zu langsam konvergiert; sehr hohe Werte kosten vor allem Rechenzeit.",
    },
    "axial_diffusion_tol": {
        "min": 1e-10,
        "max": 1e-3,
        "default": AXIAL_DIFFUSION_TOL,
        "description_en": "Stopping tolerance of the inner tissue diffusion iteration. Smaller values force a tighter tissue-field convergence before the solver accepts the current update.",
        "description_de": "Abbruchtoleranz der inneren Gewebediffusionsiteration. Kleinere Werte erzwingen eine strengere Konvergenz des Gewebefelds, bevor der aktuelle Schritt akzeptiert wird.",
    },
    "axial_coupling_max_iter": {
        "min": 2,
        "max": 80,
        "default": AXIAL_COUPLING_MAX_ITER,
        "description_en": "Maximum number of outer coupling iterations between capillary profile and tissue field. Increase it only when the coupled solution changes noticeably between iterations.",
        "description_de": "Maximale Zahl aeusserer Kopplungsiterationen zwischen Kapillarprofil und Gewebefeld. Hoehere Werte sind nur sinnvoll, wenn sich die gekoppelte Loesung zwischen den Iterationen noch deutlich aendert.",
    },
    "axial_coupling_tol": {
        "min": 1e-8,
        "max": 5e-3,
        "default": AXIAL_COUPLING_TOL,
        "description_en": "Stopping tolerance of the outer capillary-tissue coupling loop. Smaller values demand closer agreement between consecutive coupled solutions.",
        "description_de": "Abbruchtoleranz der aeusseren Kapillar-Gewebe-Kopplung. Kleinere Werte verlangen eine engere Uebereinstimmung zwischen aufeinanderfolgenden gekoppelten Loesungen.",
    },
}

# Compatibility aliases for the diagnostic MVP naming scheme
N_HILL = n_hill
P50_STD = P50

__all__ = [
    "n_hill",
    "P50",
    "C_Hb",
    "alpha",
    "PH_REF",
    "PCO2_REF",
    "TEMP_REF",
    "BOHR_COEFF",
    "CO2_COEFF",
    "TEMP_COEFF",
    "R_cap",
    "R_tis",
    "M_rate",
    "K_diff",
    "L_cap",
    "v_blood",
    "MIN_CONSUMPTION_FRACTION",
    "Q_flow",
    "NZ",
    "NR",
    "z_eval",
    "r_vec",
    "dr",
    "dz",
    "radial_weights",
    "AXIAL_DIFFUSION_MAX_ITER",
    "AXIAL_DIFFUSION_TOL",
    "AXIAL_DIFFUSION_RELAX",
    "AXIAL_COUPLING_MAX_ITER",
    "AXIAL_COUPLING_TOL",
    "CAPILLARY_ODE_RTOL",
    "CAPILLARY_ODE_ATOL",
    "CAPILLARY_ODE_MAX_STEP",
    "NUMERIC_SETTINGS_FIELDS",
    "NUMERIC_SETTINGS_TYPES",
    "NUMERIC_SETTING_SPECS",
    "N_HILL",
    "P50_STD",
]
