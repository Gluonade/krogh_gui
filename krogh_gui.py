"""Tkinter GUI for running Krogh cylinder single-case and series analyses.

Recent maintenance notes:
- Numerical solver controls for the capillary ODE and axial coupling are exposed in
    the GUI and are persisted in saved case files.
- Default capillary solver tolerances and maximum step size were tightened to reduce
    non-physical artifacts in sensitive sweeps such as mitoP50 and low-perfusion runs.
- Series runs report a sampled stricter-solver comparison so numerically sensitive
    parameter regions can be spotted without re-running the full sweep manually.
"""

import json
import os
import sys
import threading
import subprocess
from datetime import datetime
import textwrap
import tkinter as tk
from contextlib import contextmanager
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

try:
    from oxygenation_diagnostic_mvp import OxygenationInput, alert_decision
except Exception:
    OxygenationInput = None
    alert_decision = None


# -----------------------------
# Shared model constants
# -----------------------------
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

SERIES_SWEEP_FIELDS = {
    "PO2_inlet_mmHg": "P_inlet",
    "mitoP50_mmHg": "P_half",
    "pH": "pH",
    "pCO2_mmHg": "pCO2",
    "Temp_C": "temp_c",
    "Perfusion_factor": "perf",
}

# Physiologically plausible input ranges (non-blocking warnings only).
# Values outside these bounds may occur only in extreme pathological conditions
# or are entirely unrealistic for mammalian microvascular tissue.
PHYSIOLOGICAL_RANGES = {
    "PO2_inlet_mmHg":   {"warn_low": 20.0,  "warn_high": 700.0},
    "mitoP50_mmHg":     {"warn_low": 0.01,  "warn_high": 15.0},
    "pH":               {"warn_low": 6.8,   "warn_high": 7.8},
    "pCO2_mmHg":        {"warn_low": 10.0,  "warn_high": 120.0},
    "Temp_C":           {"warn_low": 20.0,  "warn_high": 43.0},
    "Perfusion_factor": {"warn_low": 0.05,  "warn_high": 30.0},
}

SERIES_RESULT_FIELDS = {
    "P50_eff": "result_p50_eff",
    "P_venous": "result_p_venous",
    "P_tissue_min": "result_p_tissue_min",
    "P_tissue_p05": "result_p_tissue_p05",
    "Hypoxic_fraction_lt1": "result_hypoxic_fraction_lt1",
    "Hypoxic_fraction_lt5": "result_hypoxic_fraction_lt5",
    "Hypoxic_fraction_lt10": "result_hypoxic_fraction_lt10",
    "PO2_fraction_gt100": "result_po2_fraction_gt_primary",
    "PO2_fraction_gt200": "result_po2_fraction_gt_secondary",
    "PO2_fraction_gt_rel1": "result_po2_fraction_gt_rel_primary",
    "PO2_fraction_gt_rel2": "result_po2_fraction_gt_rel_secondary",
    "PO2_fraction_gt_rel3": "result_po2_fraction_gt_rel_tertiary",
    "PO2_sensor_avg": "result_po2_sensor_avg",
    "S_a_percent": "result_sa",
    "S_v_percent": "result_sv",
    "Q_flow_nL_s": "result_q_flow",
}

HYPOXIC_FRACTION_FIELDS = (
    "Hypoxic_fraction_lt1",
    "Hypoxic_fraction_lt5",
    "Hypoxic_fraction_lt10",
)

INPUT_FIELD_LABELS = {
    "PO2_inlet_mmHg": "field_po2_inlet",
    "mitoP50_mmHg": "field_mitop50",
    "pH": "field_ph",
    "pCO2_mmHg": "field_pco2",
    "Temp_C": "field_temp",
    "Perfusion_factor": "field_perf",
    "High_PO2_threshold_1_mmHg": "field_high_po2_threshold_1",
    "High_PO2_threshold_2_mmHg": "field_high_po2_threshold_2",
    "High_PO2_additional_thresholds_mmHg": "field_high_po2_additional_thresholds",
    "High_PO2_relative_thresholds_percent": "field_high_po2_relative_thresholds_percent",
    "Relative_PO2_reference": "field_relative_po2_reference",
}

LANGUAGE_NAMES = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Francais",
    "it": "Italiano",
    "es": "Espanol",
}

TRANSLATIONS = {
    "en": {
        "app_title": "Krogh Model Runner",
        "language_label": "Language",
        "mode_group": "Mode",
        "mode_default": "Default script (unchanged full demonstration)",
        "mode_single": "Single-case analysis (concrete values)",
        "mode_series": "Series analysis (sweep one or two parameters)",
        "single_inputs": "Single-case inputs",
        "include_axial": "Include axial tissue diffusion",
        "series_frame": "Series analysis",
        "numerics_frame": "Numerics",
        "series_dimension": "Series type",
        "series_dimension_1d": "One varying parameter",
        "series_dimension_2d": "Two varying parameters",
        "tab_inputs": "Inputs",
        "tab_series": "Series",
        "tab_numerics": "Numerics",
        "tab_diagnostic": "Diagnostic",
        "varying_parameter": "Varying parameter",
        "start_value": "Start value",
        "end_value": "End value",
        "step_size": "Step size",
        "secondary_parameter": "Second parameter",
        "secondary_start_value": "Second start",
        "secondary_end_value": "Second end",
        "secondary_step_size": "Second step",
        "series_plot_mode": "2D/3D display",
        "series_plot_mode_2d": "2D multi-curve plot",
        "series_plot_mode_3d": "3D surface plot",
        "series_plot_mode_heatmap": "Heatmap",
        "publication_mode": "Publication mode (larger labels + high-res export)",
        "publication_layout": "Publication layout",
        "publication_layout_a4": "A4 landscape",
        "publication_layout_wide": "16:9 widescreen",
        "plot_outputs": "Plot outputs",
        "multi_select_hint": "Multiple selection with Ctrl or Shift",
        "series_selection_hint": "Only the selected outputs will be displayed after the run.",
        "save_series_results": "Save run bundle after viewing plots (Excel + plots + parameters)",
        "lock_hypoxic_fraction_scale": "Use the same y-scale for hypoxic tissue fraction plots",
        "series_plots_separate_hint": "Selected outputs are displayed in separate plot windows to keep axis labels readable.",
        "numeric_help_button": "Numerics help",
        "numeric_help_hint": "Hover over a numerics label for a short explanation or open the detailed numerics help.",
        "run_button": "Run",
        "run_diagnostic_button": "Run diagnostic",
        "save_diagnostic_template_button": "Save calibration template...",
        "reconstruct_krogh_button": "Reconstruct Krogh Cylinder",
        "diag_krogh_computing": "Fitting Krogh model to diagnostic values...",
        "diag_krogh_ready": "Krogh reconstruction ready.",
        "diag_krogh_error": "ERROR in Krogh reconstruction: {error}",
        "diag_krogh_no_result": "Run diagnostic first before reconstructing.",
        "title_3d_diagnostic": "Krogh Cylinder — {state}\nAlert: {alert} | Risk: {risk:.3f} | Confidence: {conf:.3f}\nPO2_inlet={P_inlet:.1f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f}°C\nFitted P_half={P_half:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | SensorAvg={sensor_avg:.1f} mmHg",
        "diag_krogh_fit_info": "Fitted mitoP50={P_half:.3f} mmHg to match sensor_po2={sensor_target:.1f} mmHg (simulated={sensor_sim:.1f} mmHg)",
        "plot3d_button": "3D Plot",
        "clear_button": "Clear output",
        "help_button": "Output help",
        "save_case_button": "Save case...",
        "load_case_button": "Load case...",
        "quit_button": "Quit",
        "output_group": "Output",
        "diagnostic_group": "Probabilistic oxygenation diagnostic (MVP)",
        "diag_po2": "Blood-gas po2 (mmHg)",
        "diag_pco2": "Blood-gas pco2 (mmHg)",
        "diag_ph": "pH",
        "diag_temp": "Temperature (C)",
        "diag_sensor_po2": "Sensor po2 (mmHg)",
        "diag_hemoglobin": "Hemoglobin (g/dL, optional)",
        "diag_venous_sat": "Venous saturation (%, optional)",
        "diag_optional_hint": "Note: Hemoglobin and venous saturation are optional. Leave blank to use defaults.",
        "diag_yellow_threshold": "Yellow alert threshold (low risk)",
        "diag_orange_threshold": "Orange alert threshold (elevated risk)",
        "diag_red_threshold": "Red alert threshold (critical)",
        "use_single_case_button": "Use single case values",
        "diag_result_header": "Diagnostic result",
        "diag_model_missing": "Diagnostic module oxygenation_diagnostic_mvp.py could not be loaded.",
        "diag_input_error": "Please enter valid numeric diagnostic inputs.",
        "diag_threshold_error": "Diagnostic thresholds must satisfy 0 <= yellow <= orange <= red <= 1.",
        "diag_result_line": "State={state} | risk_score={risk_score:.3f} | confidence={confidence:.3f} | certainty={certainty:.3f} | alert={alert}",
        "field_po2_inlet": "PO2_inlet_mmHg",
        "field_mitop50": "mitoP50_mmHg",
        "field_ph": "pH",
        "field_pco2": "pCO2_mmHg",
        "field_temp": "Temp_C",
        "field_perf": "Perfusion_factor",
        "field_high_po2_threshold_1": "High_PO2_threshold_1_mmHg",
        "field_high_po2_threshold_2": "High_PO2_threshold_2_mmHg",
        "field_high_po2_additional_thresholds": "Additional_high_PO2_thresholds_mmHg (comma-separated)",
        "field_high_po2_relative_thresholds_percent": "Relative_high_PO2_thresholds_percent_of_inlet (comma-separated)",
        "field_relative_po2_reference": "Relative_high_PO2_reference (inlet or tissue_max)",
        "result_p50_eff": "P50_eff (mmHg)",
        "result_p_venous": "P_venous (mmHg)",
        "result_p_tissue_min": "P_tissue_min (mmHg)",
        "result_p_tissue_p05": "P_tissue_P05 (mmHg)",
        "result_hypoxic_fraction_lt1": "Hypoxic_fraction_LT1 (%)",
        "result_hypoxic_fraction_lt5": "Hypoxic_fraction_LT5 (%)",
        "result_hypoxic_fraction_lt10": "Hypoxic_fraction_LT10 (%)",
        "result_po2_fraction_gt_primary": "PO2_fraction_GT{threshold} (%)",
        "result_po2_fraction_gt_secondary": "PO2_fraction_GT{threshold} (%)",
        "result_po2_fraction_gt_rel_primary": "PO2_fraction_GT{threshold}%_of_{reference} (%)",
        "result_po2_fraction_gt_rel_secondary": "PO2_fraction_GT{threshold}%_of_{reference} (%)",
        "result_po2_fraction_gt_rel_tertiary": "PO2_fraction_GT{threshold}%_of_{reference} (%)",
        "reference_inlet": "inlet",
        "reference_tissue_max": "tissue_max",
        "result_po2_sensor_avg": "PO2_sensor_avg (mmHg)",
        "result_sa": "S_a (%)",
        "result_sv": "S_v (%)",
        "result_q_flow": "Q_flow_local (nL/s)",
        "output_help_title": "Output parameter help",
        "output_help_intro": "Background and interpretation of the current model outputs",
        "numeric_help_title": "Numerical parameter help",
        "numeric_help_intro": "Meaning, expected effect, and recommended ranges of the adjustable solver settings",
        "numeric_help_range_line": "Recommended range: {min_value} to {max_value}. Default: {default_value}.",
        "numeric_help_current_line": "Current GUI value: {current_value}",
        "input_error_title": "Input error",
        "input_error_numeric": "Please enter valid numeric values.",
        "input_warning_threshold_order": "The second absolute high-PO2 threshold is smaller than the first one. Continue anyway?",
        "input_error_perf": "Perfusion factor must be > 0.",
        "input_error_sweep_param": "Please select a valid sweep parameter.",
        "input_error_sweep_numeric": "Please enter valid numeric sweep values.",
        "input_error_secondary_sweep_numeric": "Please enter valid numeric values for the second sweep.",
        "input_error_step_size": "Step size must be > 0.",
        "input_error_secondary_step_size": "The second step size must be > 0.",
        "input_error_perf_sweep": "Perfusion factor sweep values must be > 0.",
        "input_error_secondary_perf_sweep": "Second sweep values for the perfusion factor must be > 0.",
        "input_error_sweep_param_duplicate": "Please choose two different sweep parameters.",
        "input_error_select_result": "Please select at least one result for the series plot.",
        "input_error_numerics": "Please enter valid positive numeric solver settings.",
        "input_error_numeric_range": "{field} must be between {min_value} and {max_value}. Default: {default_value}.",
        "info_3d_title": "3D Plot",
        "info_3d_mode": "Switch to \"Single-case analysis\" mode first.",
        "save_series_title": "Save series results as...",
        "save_case_title": "Save case as...",
        "load_case_title": "Load case...",
        "save_error_title": "Save error",
        "load_error_title": "Load error",
        "json_files": "JSON files",
        "all_files": "All files",
        "excel_files": "Excel files",
        "default_not_found": "ERROR: krogh_basis.py not found.",
        "running_default": "Running default full script...\n",
        "return_code": "Return code: {code}",
        "stdout_last": "--- STDOUT (last lines) ---",
        "stderr": "--- STDERR ---",
        "default_run_error": "ERROR while running default script: {error}",
        "running_single": "Running single-case simulation...\n",
        "single_result": "Single-case result",
        "inputs_header": "  Inputs:",
        "outputs_header": "  Outputs:",
        "single_inputs_line": "    P_inlet={P_inlet:.2f} mmHg, P_MM={P_half:.2f} mmHg, pH={pH:.2f}, pCO2={pCO2:.2f} mmHg, T={temp_c:.2f} C, perf={perf:.2f}, high1={high_po2_threshold_primary:.2f} mmHg, high2={high_po2_threshold_secondary:.2f} mmHg, rel_ref={relative_po2_reference}",
        "single_run_error": "ERROR in single-case run: {error}",
        "status_ready": "Ready.",
        "status_running_default": "Running default script...",
        "status_running_single": "Running single-case analysis...",
        "status_running_series": "Running series analysis...",
        "status_series_case": "Case {index}/{count}",
        "status_series_outer": "Completed series {index}/{count} for {field} = {value:.4g}",
        "status_plotting": "Preparing plots...",
        "status_finished": "Calculation finished.",
        "status_error": "Error during calculation.",
        "running_series": "Running series simulation: {parameter} from {start:.4g} to {end:.4g} with step size {step:.4g} ({count} cases)...\n",
        "running_series_2d": "Running 2D series simulation: {parameter1} from {start1:.4g} to {end1:.4g} and {parameter2} from {start2:.4g} to {end2:.4g} ({count1} x {count2} = {count} cases)...\n",
        "series_case_progress": "  Case {index}/{count}: {parameter} = {value:.6g}",
        "series_case_progress_2d": "  Case {index}/{count}: {parameter1} = {value1:.6g}, {parameter2} = {value2:.6g}",
        "series_finished": "Series finished.",
        "results_saved": "Results saved to: {path}",
        "plot_saved": "Plot saved to: {path}",
        "save_bundle_title": "Select folder for run bundle",
        "bundle_saved": "Run bundle saved to: {path}",
        "bundle_file_parameters": "Run parameters saved to: {path}",
        "bundle_save_cancelled": "Bundle export skipped.",
        "series_results_not_saved": "Series results were not saved to Excel.",
        "series_plots_not_saved": "Series plots were displayed only and not saved to files.",
        "series_run_error": "ERROR in series run: {error}",
        "series_plot_title": "Series analysis",
        "series_plot_sweep": "Sweep: {field} = {start:.4g} to {end:.4g} (n={count})",
        "series_plot_sweep_secondary": "Second sweep: {field} = {start:.4g} to {end:.4g} (n={count})",
        "series_plot_fixed": "Fixed inputs: {params}",
        "series_plot_explanation": "Shown metric: {description}",
        "series_plot_window": "Series plot",
        "series_plot_window_field": "Series plot - {field}",
        "series_plot_surface_title": "Series surface - {field}",
        "series_plot_heatmap_title": "Series heatmap - {field}",
        "series_curve_legend": "{field} = {value:.4g}",
        "show_figures_title": "Show figures?",
        "select_figures": "Select figures to display:",
        "select_all": "Select all",
        "deselect_all": "Deselect all",
        "show_selected": "Show selected",
        "cancel": "Cancel",
        "plot3d_computing": "Computing 3D plot...",
        "plot3d_ready": "3D plot ready.",
        "plot3d_error": "ERROR in 3D plot: {error}",
        "xlabel_radial_position": "Radial position (um)",
        "ylabel_relative_length": "Relative capillary length (x)",
        "zlabel_po2": "PO2 (mmHg)",
        "legend_sensor_avg": "Sensor avg (radial mean)",
        "title_3d": "Krogh Cylinder - Single Case\nP_inlet={P_inlet:.1f} | P_MM={P_half:.2f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f} C | perf={perf:.2f}\nP50_eff={p50_eff:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | SensorAvg={sensor_avg:.1f} mmHg",
        "colorbar_po2": "PO2 (mmHg)",
        "case_saved_to": "Case saved to: {path}",
        "case_loaded_from": "Case loaded from: {path}",
        "sheet_series_results": "Series_Results",
        "sheet_series_setup": "Series_Setup",
        "column_setting": "Setting",
        "column_value": "Value",
        "setting_sweep_parameter": "Sweep parameter",
        "setting_start_value": "Start value",
        "setting_end_value": "End value",
        "setting_step_size": "Step size",
        "setting_secondary_sweep_parameter": "Second sweep parameter",
        "setting_secondary_start_value": "Second start value",
        "setting_secondary_end_value": "Second end value",
        "setting_secondary_step_size": "Second step size",
        "setting_series_dimension": "Series type",
        "setting_plot_mode": "Plot mode",
        "setting_case_count": "Case count",
        "setting_include_axial": "Include axial diffusion",
        "setting_numerics_header": "Numerics",
        "result_case": "Case",
        "result_sweep_parameter": "Sweep parameter",
        "result_sweep_value": "Sweep value",
        "result_sweep_parameter_2": "Second sweep parameter",
        "result_sweep_value_2": "Second sweep value",
        "result_include_axial": "Include axial diffusion",
        "numeric_ode_rtol": "ODE accuracy (relative)",
        "numeric_ode_atol": "ODE accuracy (absolute)",
        "numeric_ode_max_step": "Maximum ODE step along capillary (cm)",
        "numeric_axial_diffusion_max_iter": "Max iterations for tissue diffusion",
        "numeric_axial_diffusion_tol": "Tolerance for tissue diffusion",
        "numeric_axial_coupling_max_iter": "Max iterations for capillary-tissue coupling",
        "numeric_axial_coupling_tol": "Tolerance for capillary-tissue coupling",
        "series_check_header": "Numerics check on {count} sampled cases:",
        "series_check_field": "  {field}: max |delta|={abs_diff:.4g}, rel={rel_diff:.3%}, worst case #{case}",
        "series_check_ok": "  No suspicious sensitivity detected with tighter solver settings.",
        "series_check_warning": "  Warning: tighter solver settings changed at least one sampled result noticeably.",
        "physio_warning_title": "Physiological range notice",
        "physio_warning_intro": "The following input values are outside the physiologically plausible range. Such values may arise only in extreme pathological conditions or are entirely unrealistic for mammalian microvascular tissue:\n\n{warnings}\n\nContinue anyway?",
        "physio_warning_low": "  \u2022 {field} = {value:.4g} is below the plausible minimum of {limit:.4g}",
        "physio_warning_high": "  \u2022 {field} = {value:.4g} is above the plausible maximum of {limit:.4g}",
        "bool_yes": "Yes",
        "bool_no": "No",
    },
    "de": {
        "app_title": "Krogh-Modell Rechner",
        "language_label": "Sprache",
        "mode_group": "Modus",
        "mode_default": "Standardskript (unveraenderte Gesamtdemonstration)",
        "mode_single": "Einzelfallanalyse (konkrete Werte)",
        "mode_series": "Serienanalyse (ein oder zwei Parameter werden variiert)",
        "single_inputs": "Einzelfall-Eingaben",
        "include_axial": "Axiale Gewebediffusion einbeziehen",
        "series_frame": "Serienanalyse",
        "numerics_frame": "Numerik",
        "series_dimension": "Serientyp",
        "series_dimension_1d": "Ein variierter Parameter",
        "series_dimension_2d": "Zwei variierte Parameter",
        "tab_inputs": "Eingaben",
        "tab_series": "Serie",
        "tab_numerics": "Numerik",
        "tab_diagnostic": "Diagnostik",
        "varying_parameter": "Variierter Parameter",
        "start_value": "Startwert",
        "end_value": "Endwert",
        "step_size": "Schrittweite",
        "secondary_parameter": "Zweiter Parameter",
        "secondary_start_value": "Zweiter Startwert",
        "secondary_end_value": "Zweiter Endwert",
        "secondary_step_size": "Zweite Schrittweite",
        "series_plot_mode": "2D/3D-Darstellung",
        "series_plot_mode_2d": "2D-Mehrkurven-Diagramm",
        "series_plot_mode_3d": "3D-Oberflaeche",
        "series_plot_mode_heatmap": "Heatmap",
        "publication_mode": "Publikationsmodus (groessere Labels + hochaufloesender Export)",
        "publication_layout": "Publikationslayout",
        "publication_layout_a4": "A4 Querformat",
        "publication_layout_wide": "16:9 Breitbild",
        "plot_outputs": "Diagramm-Ausgaben",
        "multi_select_hint": "Mehrfachauswahl mit Strg oder Shift",
        "series_selection_hint": "Nach dem Lauf werden nur die ausgewaehlten Ergebnisse angezeigt.",
        "save_series_results": "Lauf-Buendel nach Anzeige speichern (Excel + Diagramme + Parameter)",
        "lock_hypoxic_fraction_scale": "Fuer Hypoxieanteils-Diagramme dieselbe y-Skala verwenden",
        "series_plots_separate_hint": "Ausgewaehlte Ergebnisse werden in getrennten Diagrammfenstern angezeigt, damit die Achsenbeschriftungen lesbar bleiben.",
        "numeric_help_button": "Numerikhilfe",
        "numeric_help_hint": "Mit der Maus ueber eine Numerik-Beschriftung fahren oder die ausfuehrliche Numerikhilfe oeffnen.",
        "run_button": "Starten",
        "run_diagnostic_button": "Diagnostik starten",
        "save_diagnostic_template_button": "Kalibrierungs-Vorlage speichern...",
        "reconstruct_krogh_button": "Krogh-Zylinder rekonstruieren",
        "diag_krogh_computing": "Krogh-Modell wird an Diagnostik-Werte angepasst...",
        "diag_krogh_ready": "Krogh-Rekonstruktion bereit.",
        "diag_krogh_error": "FEHLER bei Krogh-Rekonstruktion: {error}",
        "diag_krogh_no_result": "Zuerst Diagnostik ausfuehren, dann rekonstruieren.",
        "title_3d_diagnostic": "Krogh-Zylinder — {state}\nAlarm: {alert} | Risiko: {risk:.3f} | Konfidenz: {conf:.3f}\nPO2_inlet={P_inlet:.1f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f}°C\ngepasstes P_half={P_half:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | SensorMittel={sensor_avg:.1f} mmHg",
        "diag_krogh_fit_info": "Angepasstes mitoP50={P_half:.3f} mmHg fuer sensor_po2={sensor_target:.1f} mmHg (simuliert={sensor_sim:.1f} mmHg)",
        "plot3d_button": "3D-Diagramm",
        "clear_button": "Ausgabe loeschen",
        "help_button": "Parameterhilfe",
        "save_case_button": "Fall speichern...",
        "load_case_button": "Fall laden...",
        "quit_button": "Beenden",
        "output_group": "Ausgabe",
        "diagnostic_group": "Probabilistische Oxygenierungsdiagnostik (MVP)",
        "diag_po2": "Blutgas po2 (mmHg)",
        "diag_pco2": "Blutgas pco2 (mmHg)",
        "diag_ph": "pH",
        "diag_temp": "Temperatur (C)",
        "diag_sensor_po2": "Sensor po2 (mmHg)",
        "diag_hemoglobin": "Haemoglobin (g/dL, optional)",
        "diag_venous_sat": "Venoese Saettigung (%, optional)",
        "diag_optional_hint": "Hinweis: Haemoglobin und venoese Saettigung sind optional. Leer lassen fuer Standards.",
        "diag_yellow_threshold": "Gelb-Alarm-Schwelle (geringes Risiko)",
        "diag_orange_threshold": "Orange-Alarm-Schwelle (erhoehtes Risiko)",
        "diag_red_threshold": "Rot-Alarm-Schwelle (kritisch)",
        "use_single_case_button": "Werte aus Einzelfall uebernehmen",
        "diag_result_header": "Diagnostisches Ergebnis",
        "diag_model_missing": "Diagnostik-Modul oxygenation_diagnostic_mvp.py konnte nicht geladen werden.",
        "diag_input_error": "Bitte gueltige numerische Diagnostik-Eingaben eingeben.",
        "diag_threshold_error": "Diagnostik-Schwellen muessen 0 <= gelb <= orange <= rot <= 1 erfuellen.",
        "diag_result_line": "Zustand={state} | Risikowert={risk_score:.3f} | Konfidenz={confidence:.3f} | Sicherheit={certainty:.3f} | Alarm={alert}",
        "field_po2_inlet": "PO2_inlet_mmHg",
        "field_mitop50": "mitoP50_mmHg",
        "field_ph": "pH",
        "field_pco2": "pCO2_mmHg",
        "field_temp": "Temp_C",
        "field_perf": "Perfusion_factor",
        "field_high_po2_threshold_1": "Hoher_PO2_Schwellenwert_1_mmHg",
        "field_high_po2_threshold_2": "Hoher_PO2_Schwellenwert_2_mmHg",
        "field_high_po2_additional_thresholds": "Zusaetzliche_hohe_PO2_Schwellen_mmHg (kommagetrennt)",
        "field_high_po2_relative_thresholds_percent": "Relative_hohe_PO2_Schwellen_inlet_prozent (kommagetrennt)",
        "field_relative_po2_reference": "Relative_hohe_PO2_Referenz (inlet oder gewebemax)",
        "result_p50_eff": "P50_eff (mmHg)",
        "result_p_venous": "P_venous (mmHg)",
        "result_p_tissue_min": "P_tissue_min (mmHg)",
        "result_p_tissue_p05": "P_tissue_P05 (mmHg)",
        "result_hypoxic_fraction_lt1": "Hypoxieanteil_LT1 (%)",
        "result_hypoxic_fraction_lt5": "Hypoxieanteil_LT5 (%)",
        "result_hypoxic_fraction_lt10": "Hypoxieanteil_LT10 (%)",
        "result_po2_fraction_gt_primary": "PO2_Anteil_GT{threshold} (%)",
        "result_po2_fraction_gt_secondary": "PO2_Anteil_GT{threshold} (%)",
        "result_po2_fraction_gt_rel_primary": "PO2_Anteil_GT{threshold}%_von_{reference} (%)",
        "result_po2_fraction_gt_rel_secondary": "PO2_Anteil_GT{threshold}%_von_{reference} (%)",
        "result_po2_fraction_gt_rel_tertiary": "PO2_Anteil_GT{threshold}%_von_{reference} (%)",
        "reference_inlet": "Inlet",
        "reference_tissue_max": "Gewebemax",
        "result_po2_sensor_avg": "PO2_sensor_avg (mmHg)",
        "result_sa": "S_a (%)",
        "result_sv": "S_v (%)",
        "result_q_flow": "Q_flow_local (nL/s)",
        "output_help_title": "Hintergrund zu Ausgabeparametern",
        "output_help_intro": "Bedeutung und klinische Interpretation der aktuellen Modell-Ausgaben",
        "numeric_help_title": "Hintergrund zu Numerikparametern",
        "numeric_help_intro": "Bedeutung, erwartete Wirkung und empfohlene Bereiche der einstellbaren Solver-Parameter",
        "numeric_help_range_line": "Empfohlener Bereich: {min_value} bis {max_value}. Standard: {default_value}.",
        "numeric_help_current_line": "Aktueller GUI-Wert: {current_value}",
        "input_error_title": "Eingabefehler",
        "input_error_numeric": "Bitte gueltige numerische Werte eingeben.",
        "input_warning_threshold_order": "Der zweite absolute hohe PO2-Schwellenwert ist kleiner als der erste. Trotzdem fortfahren?",
        "input_error_perf": "Der Perfusionsfaktor muss > 0 sein.",
        "input_error_sweep_param": "Bitte einen gueltigen Sweep-Parameter auswaehlen.",
        "input_error_sweep_numeric": "Bitte gueltige numerische Sweep-Werte eingeben.",
        "input_error_secondary_sweep_numeric": "Bitte gueltige numerische Werte fuer den zweiten Sweep eingeben.",
        "input_error_step_size": "Die Schrittweite muss > 0 sein.",
        "input_error_secondary_step_size": "Die zweite Schrittweite muss > 0 sein.",
        "input_error_perf_sweep": "Sweep-Werte fuer den Perfusionsfaktor muessen > 0 sein.",
        "input_error_secondary_perf_sweep": "Werte des zweiten Sweeps fuer den Perfusionsfaktor muessen > 0 sein.",
        "input_error_sweep_param_duplicate": "Bitte zwei unterschiedliche Sweep-Parameter auswaehlen.",
        "input_error_select_result": "Bitte mindestens ein Ergebnis fuer das Serien-Diagramm auswaehlen.",
        "input_error_numerics": "Bitte gueltige positive numerische Solver-Einstellungen eingeben.",
        "input_error_numeric_range": "{field} muss zwischen {min_value} und {max_value} liegen. Standard: {default_value}.",
        "info_3d_title": "3D-Diagramm",
        "info_3d_mode": "Bitte zuerst in den Modus \"Einzelfallanalyse\" wechseln.",
        "save_series_title": "Serienergebnisse speichern unter...",
        "save_case_title": "Fall speichern unter...",
        "load_case_title": "Fall laden...",
        "save_error_title": "Speicherfehler",
        "load_error_title": "Ladefehler",
        "json_files": "JSON-Dateien",
        "all_files": "Alle Dateien",
        "excel_files": "Excel-Dateien",
        "default_not_found": "FEHLER: krogh_basis.py wurde nicht gefunden.",
        "running_default": "Standardskript wird ausgefuehrt...\n",
        "return_code": "Rueckgabecode: {code}",
        "stdout_last": "--- STDOUT (letzte Zeilen) ---",
        "stderr": "--- STDERR ---",
        "default_run_error": "FEHLER beim Ausfuehren des Standardskripts: {error}",
        "running_single": "Einzelfall-Simulation wird ausgefuehrt...\n",
        "single_result": "Ergebnis der Einzelfallanalyse",
        "inputs_header": "  Eingaben:",
        "outputs_header": "  Ausgaben:",
        "single_inputs_line": "    P_inlet={P_inlet:.2f} mmHg, P_MM={P_half:.2f} mmHg, pH={pH:.2f}, pCO2={pCO2:.2f} mmHg, T={temp_c:.2f} C, Perfusion={perf:.2f}, Schwelle1={high_po2_threshold_primary:.2f} mmHg, Schwelle2={high_po2_threshold_secondary:.2f} mmHg, rel_ref={relative_po2_reference}",
        "single_run_error": "FEHLER in der Einzelfallanalyse: {error}",
        "status_ready": "Bereit.",
        "status_running_default": "Standardskript wird ausgefuehrt...",
        "status_running_single": "Einzelfallanalyse wird ausgefuehrt...",
        "status_running_series": "Serienanalyse wird ausgefuehrt...",
        "status_series_case": "Fall {index}/{count}",
        "status_series_outer": "Teilserie {index}/{count} abgeschlossen fuer {field} = {value:.4g}",
        "status_plotting": "Diagramme werden vorbereitet...",
        "status_finished": "Berechnung abgeschlossen.",
        "status_error": "Fehler waehrend der Berechnung.",
        "running_series": "Serienanalyse wird ausgefuehrt: {parameter} von {start:.4g} bis {end:.4g} mit Schrittweite {step:.4g} ({count} Faelle)...\n",
        "running_series_2d": "2D-Serienanalyse wird ausgefuehrt: {parameter1} von {start1:.4g} bis {end1:.4g} und {parameter2} von {start2:.4g} bis {end2:.4g} ({count1} x {count2} = {count} Faelle)...\n",
        "series_case_progress": "  Fall {index}/{count}: {parameter} = {value:.6g}",
        "series_case_progress_2d": "  Fall {index}/{count}: {parameter1} = {value1:.6g}, {parameter2} = {value2:.6g}",
        "series_finished": "Serienanalyse abgeschlossen.",
        "results_saved": "Ergebnisse gespeichert unter: {path}",
        "plot_saved": "Diagramm gespeichert unter: {path}",
        "save_bundle_title": "Ordner fuer Lauf-Buendel auswaehlen",
        "bundle_saved": "Lauf-Buendel gespeichert unter: {path}",
        "bundle_file_parameters": "Laufparameter gespeichert unter: {path}",
        "bundle_save_cancelled": "Buendel-Export uebersprungen.",
        "series_results_not_saved": "Serienergebnisse wurden nicht als Excel-Datei gespeichert.",
        "series_plots_not_saved": "Seriendiagramme wurden nur angezeigt und nicht als Dateien gespeichert.",
        "series_run_error": "FEHLER in der Serienanalyse: {error}",
        "series_plot_title": "Serienanalyse",
        "series_plot_sweep": "Sweep: {field} = {start:.4g} bis {end:.4g} (n={count})",
        "series_plot_sweep_secondary": "Zweiter Sweep: {field} = {start:.4g} bis {end:.4g} (n={count})",
        "series_plot_fixed": "Feste Eingaben: {params}",
        "series_plot_explanation": "Dargestellter Parameter: {description}",
        "series_plot_window": "Seriendiagramm",
        "series_plot_window_field": "Seriendiagramm - {field}",
        "series_plot_surface_title": "Serienoberflaeche - {field}",
        "series_plot_heatmap_title": "Serien-Heatmap - {field}",
        "series_curve_legend": "{field} = {value:.4g}",
        "show_figures_title": "Abbildungen anzeigen?",
        "select_figures": "Auszugebende Abbildungen auswaehlen:",
        "select_all": "Alle auswaehlen",
        "deselect_all": "Auswahl aufheben",
        "show_selected": "Auswahl anzeigen",
        "cancel": "Abbrechen",
        "plot3d_computing": "3D-Diagramm wird berechnet...",
        "plot3d_ready": "3D-Diagramm ist bereit.",
        "plot3d_error": "FEHLER im 3D-Diagramm: {error}",
        "xlabel_radial_position": "Radiale Position (um)",
        "ylabel_relative_length": "Relative Kapillarlaenge (x)",
        "zlabel_po2": "PO2 (mmHg)",
        "legend_sensor_avg": "Sensor-Mittelwert (radialer Mittelwert)",
        "title_3d": "Krogh-Zylinder - Einzelfall\nP_inlet={P_inlet:.1f} | P_MM={P_half:.2f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f} C | Perfusion={perf:.2f}\nP50_eff={p50_eff:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | SensorMittel={sensor_avg:.1f} mmHg",
        "colorbar_po2": "PO2 (mmHg)",
        "case_saved_to": "Fall gespeichert unter: {path}",
        "case_loaded_from": "Fall geladen aus: {path}",
        "sheet_series_results": "Serien_Ergebnisse",
        "sheet_series_setup": "Serien_Setup",
        "column_setting": "Einstellung",
        "column_value": "Wert",
        "setting_sweep_parameter": "Sweep-Parameter",
        "setting_start_value": "Startwert",
        "setting_end_value": "Endwert",
        "setting_step_size": "Schrittweite",
        "setting_secondary_sweep_parameter": "Zweiter Sweep-Parameter",
        "setting_secondary_start_value": "Zweiter Startwert",
        "setting_secondary_end_value": "Zweiter Endwert",
        "setting_secondary_step_size": "Zweite Schrittweite",
        "setting_series_dimension": "Serientyp",
        "setting_plot_mode": "Diagrammtyp",
        "setting_case_count": "Anzahl Faelle",
        "setting_include_axial": "Axiale Diffusion einbeziehen",
        "setting_numerics_header": "Numerik",
        "result_case": "Fall",
        "result_sweep_parameter": "Sweep-Parameter",
        "result_sweep_value": "Sweep-Wert",
        "result_sweep_parameter_2": "Zweiter Sweep-Parameter",
        "result_sweep_value_2": "Zweiter Sweep-Wert",
        "result_include_axial": "Axiale Diffusion einbeziehen",
        "numeric_ode_rtol": "ODE-Genauigkeit (relativ)",
        "numeric_ode_atol": "ODE-Genauigkeit (absolut)",
        "numeric_ode_max_step": "Maximaler ODE-Schritt entlang der Kapillare (cm)",
        "numeric_axial_diffusion_max_iter": "Maximale Iterationen fuer Gewebediffusion",
        "numeric_axial_diffusion_tol": "Toleranz fuer Gewebediffusion",
        "numeric_axial_coupling_max_iter": "Maximale Iterationen fuer Kapillar-Gewebe-Kopplung",
        "numeric_axial_coupling_tol": "Toleranz fuer Kapillar-Gewebe-Kopplung",
        "series_check_header": "Numerik-Check mit {count} Stichprobenfaellen:",
        "series_check_field": "  {field}: max |Delta|={abs_diff:.4g}, rel={rel_diff:.3%}, schlechtester Fall #{case}",
        "series_check_ok": "  Keine auffaellige Sensitivitaet mit strengeren Solver-Einstellungen erkannt.",
        "series_check_warning": "  Warnung: Strengere Solver-Einstellungen haben mindestens ein Stichprobenergebnis merklich veraendert.",
        "physio_warning_title": "Hinweis: Physiologischer Bereich",
        "physio_warning_intro": "Die folgenden Eingabewerte liegen ausserhalb des physiologisch plausiblen Bereichs. Diese Werte koennen moeglicherweise nur unter extremen pathologischen Bedingungen auftreten oder sind fuer saeugetierisches Mikrozirkulationsgewebe gaenzlich unrealistisch:\n\n{warnings}\n\nTrotzdem fortfahren?",
        "physio_warning_low": "  \u2022 {field} = {value:.4g} liegt unter dem plausiblen Mindestwert von {limit:.4g}",
        "physio_warning_high": "  \u2022 {field} = {value:.4g} liegt ueber dem plausiblen Hoechstwert von {limit:.4g}",
        "bool_yes": "Ja",
        "bool_no": "Nein",
    },
    "fr": {
        "app_title": "Calculateur du modele de Krogh",
        "language_label": "Langue",
        "mode_group": "Mode",
        "mode_default": "Script par defaut (demonstration complete inchangee)",
        "mode_single": "Analyse mono-cas (valeurs concretes)",
        "mode_series": "Analyse en serie (variation d'un parametre)",
        "single_inputs": "Entrees du cas unique",
        "include_axial": "Inclure la diffusion tissulaire axiale",
        "series_frame": "Analyse en serie",
        "varying_parameter": "Parametre varie",
        "start_value": "Valeur initiale",
        "end_value": "Valeur finale",
        "step_size": "Pas",
        "plot_outputs": "Resultats a tracer",
        "multi_select_hint": "Selection multiple avec Ctrl ou Shift",
        "run_button": "Executer",
        "plot3d_button": "Graphique 3D",
        "clear_button": "Effacer la sortie",
        "help_button": "Aide sorties",
        "save_case_button": "Enregistrer le cas...",
        "load_case_button": "Charger le cas...",
        "quit_button": "Quitter",
        "output_group": "Sortie",
        "field_po2_inlet": "PO2_inlet_mmHg",
        "field_mitop50": "mitoP50_mmHg",
        "field_ph": "pH",
        "field_pco2": "pCO2_mmHg",
        "field_temp": "Temp_C",
        "field_perf": "Perfusion_factor",
        "field_high_po2_threshold_1": "Seuil_haute_PO2_1_mmHg",
        "field_high_po2_threshold_2": "Seuil_haute_PO2_2_mmHg",
        "field_high_po2_additional_thresholds": "Seuils_additionnels_haute_PO2_mmHg (separes par virgules)",
        "field_high_po2_relative_thresholds_percent": "Seuils_haute_PO2_relatifs_pourcent_entree (separes par virgules)",
        "field_relative_po2_reference": "Reference_haute_PO2_relative (entree ou max_tissu)",
        "result_p50_eff": "P50_eff (mmHg)",
        "result_p_venous": "P_venous (mmHg)",
        "result_p_tissue_min": "P_tissue_min (mmHg)",
        "result_p_tissue_p05": "P_tissue_P05 (mmHg)",
        "result_hypoxic_fraction_lt1": "Fraction_hypoxique_LT1 (%)",
        "result_hypoxic_fraction_lt5": "Fraction_hypoxique_LT5 (%)",
        "result_hypoxic_fraction_lt10": "Fraction_hypoxique_LT10 (%)",
        "result_po2_fraction_gt_primary": "Fraction_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_secondary": "Fraction_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_rel_primary": "Fraction_PO2_GT{threshold}%_de_{reference} (%)",
        "result_po2_fraction_gt_rel_secondary": "Fraction_PO2_GT{threshold}%_de_{reference} (%)",
        "result_po2_fraction_gt_rel_tertiary": "Fraction_PO2_GT{threshold}%_de_{reference} (%)",
        "reference_inlet": "entree",
        "reference_tissue_max": "max_tissu",
        "result_po2_sensor_avg": "PO2_sensor_avg (mmHg)",
        "result_sa": "S_a (%)",
        "result_sv": "S_v (%)",
        "result_q_flow": "Q_flow_local (nL/s)",
        "output_help_title": "Aide des parametres de sortie",
        "output_help_intro": "Signification et interpretation des sorties actuelles du modele",
        "input_error_title": "Erreur de saisie",
        "input_error_numeric": "Veuillez saisir des valeurs numeriques valides.",
        "input_error_perf": "Le facteur de perfusion doit etre > 0.",
        "input_error_sweep_param": "Veuillez selectionner un parametre de balayage valide.",
        "input_error_sweep_numeric": "Veuillez saisir des valeurs de balayage numeriques valides.",
        "input_error_step_size": "Le pas doit etre > 0.",
        "input_error_perf_sweep": "Les valeurs de balayage du facteur de perfusion doivent etre > 0.",
        "input_error_select_result": "Veuillez selectionner au moins un resultat pour le graphique de serie.",
        "info_3d_title": "Graphique 3D",
        "info_3d_mode": "Passez d'abord au mode \"Analyse mono-cas\".",
        "save_series_results": "Enregistrer le lot d'execution apres affichage (Excel + graphiques + parametres)",
        "save_series_title": "Enregistrer les resultats de serie sous...",
        "save_case_title": "Enregistrer le cas sous...",
        "load_case_title": "Charger le cas...",
        "save_error_title": "Erreur d'enregistrement",
        "load_error_title": "Erreur de chargement",
        "json_files": "Fichiers JSON",
        "all_files": "Tous les fichiers",
        "excel_files": "Fichiers Excel",
        "default_not_found": "ERREUR : krogh_basis.py est introuvable.",
        "running_default": "Execution du script par defaut...\n",
        "return_code": "Code de retour : {code}",
        "stdout_last": "--- STDOUT (dernieres lignes) ---",
        "stderr": "--- STDERR ---",
        "default_run_error": "ERREUR lors de l'execution du script par defaut : {error}",
        "running_single": "Execution de la simulation mono-cas...\n",
        "single_result": "Resultat de l'analyse mono-cas",
        "inputs_header": "  Entrees :",
        "outputs_header": "  Sorties :",
        "single_inputs_line": "    P_inlet={P_inlet:.2f} mmHg, P_MM={P_half:.2f} mmHg, pH={pH:.2f}, pCO2={pCO2:.2f} mmHg, T={temp_c:.2f} C, perf={perf:.2f}, high1={high_po2_threshold_primary:.2f} mmHg, high2={high_po2_threshold_secondary:.2f} mmHg",
        "single_run_error": "ERREUR dans l'analyse mono-cas : {error}",
        "running_series": "Execution de l'analyse en serie : {parameter} de {start:.4g} a {end:.4g} avec un pas de {step:.4g} ({count} cas)...\n",
        "series_case_progress": "  Cas {index}/{count} : {parameter} = {value:.6g}",
        "series_finished": "Analyse en serie terminee.",
        "results_saved": "Resultats enregistres dans : {path}",
        "plot_saved": "Graphique enregistre dans : {path}",
        "save_bundle_title": "Selectionner le dossier du lot d'execution",
        "bundle_saved": "Lot d'execution enregistre dans : {path}",
        "bundle_file_parameters": "Parametres d'execution enregistres dans : {path}",
        "bundle_save_cancelled": "Export du lot ignore.",
        "series_run_error": "ERREUR dans l'analyse en serie : {error}",
        "series_plot_title": "Analyse en serie",
        "publication_mode": "Mode publication (libelles plus grands + export haute resolution)",
        "publication_layout": "Mise en page publication",
        "publication_layout_a4": "A4 paysage",
        "publication_layout_wide": "16:9 ecran large",
        "series_plot_sweep": "Balayage : {field} = {start:.4g} a {end:.4g} (n={count})",
        "series_plot_fixed": "Entrees fixes : {params}",
        "series_plot_window": "Graphique de serie",
        "show_figures_title": "Afficher les figures ?",
        "select_figures": "Selectionnez les figures a afficher :",
        "select_all": "Tout selectionner",
        "deselect_all": "Tout deselectionner",
        "show_selected": "Afficher la selection",
        "cancel": "Annuler",
        "plot3d_computing": "Calcul du graphique 3D...",
        "plot3d_ready": "Graphique 3D pret.",
        "plot3d_error": "ERREUR dans le graphique 3D : {error}",
        "xlabel_radial_position": "Position radiale (um)",
        "ylabel_relative_length": "Longueur capillaire relative (x)",
        "zlabel_po2": "PO2 (mmHg)",
        "legend_sensor_avg": "Moyenne capteur (moyenne radiale)",
        "title_3d": "Cylindre de Krogh - Cas unique\nP_inlet={P_inlet:.1f} | P_MM={P_half:.2f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f} C | perf={perf:.2f}\nP50_eff={p50_eff:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | MoyCapteur={sensor_avg:.1f} mmHg",
        "colorbar_po2": "PO2 (mmHg)",
        "case_saved_to": "Cas enregistre dans : {path}",
        "case_loaded_from": "Cas charge depuis : {path}",
        "sheet_series_results": "Resultats_Serie",
        "sheet_series_setup": "Parametres_Serie",
        "column_setting": "Parametre",
        "column_value": "Valeur",
        "setting_sweep_parameter": "Parametre varie",
        "setting_start_value": "Valeur initiale",
        "setting_end_value": "Valeur finale",
        "setting_step_size": "Pas",
        "setting_case_count": "Nombre de cas",
        "setting_include_axial": "Inclure la diffusion axiale",
        "result_case": "Cas",
        "result_sweep_parameter": "Parametre varie",
        "result_sweep_value": "Valeur de balayage",
        "result_include_axial": "Inclure la diffusion axiale",
        "physio_warning_title": "Avertissement: plage physiologique",
        "physio_warning_intro": "Les valeurs d'entree suivantes sont en dehors de la plage physiologiquement plausible. De telles valeurs ne peuvent apparaitre que dans des conditions pathologiques extremes ou sont completement irrealistes pour les tissus microvasculaires des mammiferes :\n\n{warnings}\n\nContinuer quand meme ?",
        "physio_warning_low": "  \u2022 {field} = {value:.4g} est en dessous du minimum plausible de {limit:.4g}",
        "physio_warning_high": "  \u2022 {field} = {value:.4g} est au-dessus du maximum plausible de {limit:.4g}",
        "bool_yes": "Oui",
        "bool_no": "Non",
        "tab_diagnostic": "Diagnostic",
        "run_diagnostic_button": "Executer diagnostic",
        "save_diagnostic_template_button": "Enregistrer modele etalonnage...",
        "reconstruct_krogh_button": "Reconstruire cylindre de Krogh",
        "diag_krogh_computing": "Ajustement du modele de Krogh aux valeurs diagnostiques...",
        "diag_krogh_ready": "Reconstruction de Krogh prete.",
        "diag_krogh_error": "ERREUR reconstruction Krogh : {error}",
        "diag_krogh_no_result": "Executer d'abord le diagnostic avant de reconstruire.",
        "title_3d_diagnostic": "Cylindre de Krogh — {state}\nAlerte: {alert} | Risque: {risk:.3f} | Confiance: {conf:.3f}\nPO2_inlet={P_inlet:.1f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f}°C\nP_half ajuste={P_half:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | MoyCapteur={sensor_avg:.1f} mmHg",
        "diag_krogh_fit_info": "P_half ajuste={P_half:.3f} mmHg pour sensor_po2={sensor_target:.1f} mmHg (simule={sensor_sim:.1f} mmHg)",
        "diagnostic_group": "Diagnostic probabiliste d'oxygenation (MVP)",
        "diag_po2": "Gaz du sang po2 (mmHg)",
        "diag_pco2": "Gaz du sang pco2 (mmHg)",
        "diag_ph": "pH",
        "diag_temp": "Temperature (C)",
        "diag_sensor_po2": "Capteur po2 (mmHg)",
        "diag_hemoglobin": "Hemoglobine (g/dL, optionnel)",
        "diag_venous_sat": "Saturation veineuse (%, optionnel)",
        "diag_optional_hint": "Remarque : l'hemoglobine et la saturation veineuse sont optionnelles. Laissez vide pour utiliser les valeurs par defaut.",
        "diag_yellow_threshold": "Seuil alerte jaune (faible risque)",
        "diag_orange_threshold": "Seuil alerte orange (risque eleve)",
        "diag_red_threshold": "Seuil alerte rouge (critique)",
        "use_single_case_button": "Utiliser valeurs cas unique",
        "diag_result_header": "Resultat du diagnostic",
        "diag_model_missing": "Impossible de charger le module diagnostic oxygenation_diagnostic_mvp.py.",
        "diag_input_error": "Veuillez entrer des valeurs numeriques valides pour le diagnostic.",
        "diag_threshold_error": "Les seuils diagnostic doivent satisfaire 0 <= jaune <= orange <= rouge <= 1.",
        "diag_result_line": "Etat={state} | score_risque={risk_score:.3f} | confiance={confidence:.3f} | certitude={certainty:.3f} | alerte={alert}",
    },
    "it": {
        "app_title": "Calcolatore del modello di Krogh",
        "language_label": "Lingua",
        "mode_group": "Modalita",
        "mode_default": "Script predefinito (dimostrazione completa invariata)",
        "mode_single": "Analisi del caso singolo (valori concreti)",
        "mode_series": "Analisi di serie (variazione di un parametro)",
        "single_inputs": "Input del caso singolo",
        "include_axial": "Includi la diffusione tissutale assiale",
        "series_frame": "Analisi di serie",
        "varying_parameter": "Parametro variato",
        "start_value": "Valore iniziale",
        "end_value": "Valore finale",
        "step_size": "Passo",
        "plot_outputs": "Risultati da tracciare",
        "multi_select_hint": "Selezione multipla con Ctrl o Shift",
        "run_button": "Esegui",
        "plot3d_button": "Grafico 3D",
        "clear_button": "Cancella output",
        "help_button": "Aiuto output",
        "save_case_button": "Salva caso...",
        "load_case_button": "Carica caso...",
        "quit_button": "Esci",
        "output_group": "Output",
        "field_po2_inlet": "PO2_inlet_mmHg",
        "field_mitop50": "mitoP50_mmHg",
        "field_ph": "pH",
        "field_pco2": "pCO2_mmHg",
        "field_temp": "Temp_C",
        "field_perf": "Perfusion_factor",
        "field_high_po2_threshold_1": "Soglia_PO2_alta_1_mmHg",
        "field_high_po2_threshold_2": "Soglia_PO2_alta_2_mmHg",
        "field_high_po2_additional_thresholds": "Soglie_ulteriori_PO2_alta_mmHg (separate da virgole)",
        "field_high_po2_relative_thresholds_percent": "Soglie_PO2_alta_relative_percentuale_ingresso (separate da virgole)",
        "field_relative_po2_reference": "Riferimento_PO2_alta_relativo (ingresso o max_tessuto)",
        "result_p50_eff": "P50_eff (mmHg)",
        "result_p_venous": "P_venous (mmHg)",
        "result_p_tissue_min": "P_tissue_min (mmHg)",
        "result_p_tissue_p05": "P_tissue_P05 (mmHg)",
        "result_hypoxic_fraction_lt1": "Frazione_ipossica_LT1 (%)",
        "result_hypoxic_fraction_lt5": "Frazione_ipossica_LT5 (%)",
        "result_hypoxic_fraction_lt10": "Frazione_ipossica_LT10 (%)",
        "result_po2_fraction_gt_primary": "Frazione_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_secondary": "Frazione_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_rel_primary": "Frazione_PO2_GT{threshold}%_di_{reference} (%)",
        "result_po2_fraction_gt_rel_secondary": "Frazione_PO2_GT{threshold}%_di_{reference} (%)",
        "result_po2_fraction_gt_rel_tertiary": "Frazione_PO2_GT{threshold}%_di_{reference} (%)",
        "reference_inlet": "ingresso",
        "reference_tissue_max": "max_tessuto",
        "result_po2_sensor_avg": "PO2_sensor_avg (mmHg)",
        "result_sa": "S_a (%)",
        "result_sv": "S_v (%)",
        "result_q_flow": "Q_flow_local (nL/s)",
        "output_help_title": "Guida ai parametri di output",
        "output_help_intro": "Significato e interpretazione degli output attuali del modello",
        "input_error_title": "Errore di input",
        "input_error_numeric": "Inserire valori numerici validi.",
        "input_error_perf": "Il fattore di perfusione deve essere > 0.",
        "input_error_sweep_param": "Selezionare un parametro di sweep valido.",
        "input_error_sweep_numeric": "Inserire valori numerici validi per lo sweep.",
        "input_error_step_size": "Il passo deve essere > 0.",
        "input_error_perf_sweep": "I valori di sweep del fattore di perfusione devono essere > 0.",
        "input_error_select_result": "Selezionare almeno un risultato per il grafico di serie.",
        "info_3d_title": "Grafico 3D",
        "info_3d_mode": "Passare prima alla modalita \"Analisi del caso singolo\".",
        "save_series_results": "Salva il bundle esecuzione dopo la visualizzazione (Excel + grafici + parametri)",
        "save_series_title": "Salva i risultati della serie come...",
        "save_case_title": "Salva caso come...",
        "load_case_title": "Carica caso...",
        "save_error_title": "Errore di salvataggio",
        "load_error_title": "Errore di caricamento",
        "json_files": "File JSON",
        "all_files": "Tutti i file",
        "excel_files": "File Excel",
        "default_not_found": "ERRORE: krogh_basis.py non trovato.",
        "running_default": "Esecuzione dello script predefinito...\n",
        "return_code": "Codice di ritorno: {code}",
        "stdout_last": "--- STDOUT (ultime righe) ---",
        "stderr": "--- STDERR ---",
        "default_run_error": "ERRORE durante l'esecuzione dello script predefinito: {error}",
        "running_single": "Esecuzione della simulazione del caso singolo...\n",
        "single_result": "Risultato del caso singolo",
        "inputs_header": "  Input:",
        "outputs_header": "  Output:",
        "single_inputs_line": "    P_inlet={P_inlet:.2f} mmHg, P_MM={P_half:.2f} mmHg, pH={pH:.2f}, pCO2={pCO2:.2f} mmHg, T={temp_c:.2f} C, perf={perf:.2f}, high1={high_po2_threshold_primary:.2f} mmHg, high2={high_po2_threshold_secondary:.2f} mmHg",
        "single_run_error": "ERRORE nell'analisi del caso singolo: {error}",
        "running_series": "Esecuzione dell'analisi di serie: {parameter} da {start:.4g} a {end:.4g} con passo {step:.4g} ({count} casi)...\n",
        "series_case_progress": "  Caso {index}/{count}: {parameter} = {value:.6g}",
        "series_finished": "Analisi di serie completata.",
        "results_saved": "Risultati salvati in: {path}",
        "plot_saved": "Grafico salvato in: {path}",
        "save_bundle_title": "Seleziona la cartella per il bundle di esecuzione",
        "bundle_saved": "Bundle di esecuzione salvato in: {path}",
        "bundle_file_parameters": "Parametri di esecuzione salvati in: {path}",
        "bundle_save_cancelled": "Esportazione bundle annullata.",
        "series_run_error": "ERRORE nell'analisi di serie: {error}",
        "series_plot_title": "Analisi di serie",
        "publication_mode": "Modalita pubblicazione (etichette piu grandi + export ad alta risoluzione)",
        "publication_layout": "Layout pubblicazione",
        "publication_layout_a4": "A4 orizzontale",
        "publication_layout_wide": "16:9 widescreen",
        "series_plot_sweep": "Sweep: {field} = {start:.4g} a {end:.4g} (n={count})",
        "series_plot_fixed": "Ingressi fissi: {params}",
        "series_plot_window": "Grafico di serie",
        "show_figures_title": "Mostrare le figure?",
        "select_figures": "Selezionare le figure da mostrare:",
        "select_all": "Seleziona tutto",
        "deselect_all": "Deseleziona tutto",
        "show_selected": "Mostra selezione",
        "cancel": "Annulla",
        "plot3d_computing": "Calcolo del grafico 3D...",
        "plot3d_ready": "Grafico 3D pronto.",
        "plot3d_error": "ERRORE nel grafico 3D: {error}",
        "xlabel_radial_position": "Posizione radiale (um)",
        "ylabel_relative_length": "Lunghezza capillare relativa (x)",
        "zlabel_po2": "PO2 (mmHg)",
        "legend_sensor_avg": "Media sensore (media radiale)",
        "title_3d": "Cilindro di Krogh - Caso singolo\nP_inlet={P_inlet:.1f} | P_MM={P_half:.2f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f} C | perf={perf:.2f}\nP50_eff={p50_eff:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | MediaSensore={sensor_avg:.1f} mmHg",
        "colorbar_po2": "PO2 (mmHg)",
        "case_saved_to": "Caso salvato in: {path}",
        "case_loaded_from": "Caso caricato da: {path}",
        "sheet_series_results": "Risultati_Serie",
        "sheet_series_setup": "Impostazioni_Serie",
        "column_setting": "Impostazione",
        "column_value": "Valore",
        "setting_sweep_parameter": "Parametro variato",
        "setting_start_value": "Valore iniziale",
        "setting_end_value": "Valore finale",
        "setting_step_size": "Passo",
        "setting_case_count": "Numero di casi",
        "setting_include_axial": "Includi diffusione assiale",
        "result_case": "Caso",
        "result_sweep_parameter": "Parametro variato",
        "result_sweep_value": "Valore sweep",
        "result_include_axial": "Includi diffusione assiale",
        "physio_warning_title": "Avviso: intervallo fisiologico",
        "physio_warning_intro": "I seguenti valori di input sono al di fuori dell'intervallo fisiologicamente plausibile. Tali valori possono verificarsi solo in condizioni patologiche estreme o sono del tutto irrealistici per i tessuti microvascolari dei mammiferi:\n\n{warnings}\n\nContinuare comunque?",
        "physio_warning_low": "  \u2022 {field} = {value:.4g} e' al di sotto del minimo plausibile di {limit:.4g}",
        "physio_warning_high": "  \u2022 {field} = {value:.4g} e' al di sopra del massimo plausibile di {limit:.4g}",
        "bool_yes": "Si",
        "bool_no": "No",
        "tab_diagnostic": "Diagnostica",
        "diagnostic_group": "Diagnostica probabilistica dell'ossigenazione (MVP)",
        "diag_po2": "Gas del sangue po2 (mmHg)",
        "diag_pco2": "Gas del sangue pco2 (mmHg)",
        "diag_ph": "pH",
        "diag_temp": "Temperatura (C)",
        "diag_sensor_po2": "Sensore po2 (mmHg)",
        "diag_hemoglobin": "Emoglobina (g/dL, opzionale)",
        "diag_venous_sat": "Saturazione venosa (%, opzionale)",
        "diag_optional_hint": "Nota: l'emoglobina e la saturazione venosa sono opzionali. Lascia vuoto per usare i valori predefiniti.",
        "diag_yellow_threshold": "Soglia allerta gialla (rischio basso)",
        "diag_orange_threshold": "Soglia allerta arancione (rischio elevato)",
        "diag_red_threshold": "Soglia allerta rossa (critico)",
        "use_single_case_button": "Usa valori del caso singolo",
        "diag_result_header": "Risultato della diagnostica",
        "diag_model_missing": "Impossibile caricare il modulo diagnostico oxygenation_diagnostic_mvp.py.",
        "diag_input_error": "Inserisci valori numerici validi per la diagnostica.",
        "diag_threshold_error": "Le soglie diagnostiche devono soddisfare 0 <= giallo <= arancione <= rosso <= 1.",
        "diag_result_line": "Stato={state} | punteggio_rischio={risk_score:.3f} | confidenza={confidence:.3f} | certezza={certainty:.3f} | allarme={alert}",
        "run_diagnostic_button": "Esegui diagnostica",
        "save_diagnostic_template_button": "Salva modello di calibrazione...",
        "reconstruct_krogh_button": "Ricostruisci cilindro di Krogh",
        "diag_krogh_computing": "Adattamento modello di Krogh ai valori diagnostici...",
        "diag_krogh_ready": "Ricostruzione di Krogh pronta.",
        "diag_krogh_error": "ERRORE nella ricostruzione di Krogh: {error}",
        "diag_krogh_no_result": "Eseguire prima la diagnostica prima di ricostruire.",
        "title_3d_diagnostic": "Cilindro di Krogh — {state}\nAllarme: {alert} | Rischio: {risk:.3f} | Confidenza: {conf:.3f}\nPO2_inlet={P_inlet:.1f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f}°C\nP_half adattato={P_half:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | MediaSensore={sensor_avg:.1f} mmHg",
        "diag_krogh_fit_info": "P_half adattato={P_half:.3f} mmHg per sensor_po2={sensor_target:.1f} mmHg (simulato={sensor_sim:.1f} mmHg)",
    },
    "es": {
        "app_title": "Calculadora del modelo de Krogh",
        "language_label": "Idioma",
        "mode_group": "Modo",
        "mode_default": "Script predeterminado (demostracion completa sin cambios)",
        "mode_single": "Analisis de caso unico (valores concretos)",
        "mode_series": "Analisis en serie (variacion de un parametro)",
        "single_inputs": "Entradas de caso unico",
        "include_axial": "Incluir difusion tisular axial",
        "series_frame": "Analisis en serie",
        "varying_parameter": "Parametro variado",
        "start_value": "Valor inicial",
        "end_value": "Valor final",
        "step_size": "Paso",
        "plot_outputs": "Resultados para graficar",
        "multi_select_hint": "Seleccion multiple con Ctrl o Shift",
        "run_button": "Ejecutar",
        "plot3d_button": "Grafico 3D",
        "clear_button": "Borrar salida",
        "help_button": "Ayuda de salida",
        "save_case_button": "Guardar caso...",
        "load_case_button": "Cargar caso...",
        "quit_button": "Salir",
        "output_group": "Salida",
        "field_po2_inlet": "PO2_inlet_mmHg",
        "field_mitop50": "mitoP50_mmHg",
        "field_ph": "pH",
        "field_pco2": "pCO2_mmHg",
        "field_temp": "Temp_C",
        "field_perf": "Perfusion_factor",
        "field_high_po2_threshold_1": "Umbral_PO2_alto_1_mmHg",
        "field_high_po2_threshold_2": "Umbral_PO2_alto_2_mmHg",
        "field_high_po2_additional_thresholds": "Umbrales_adicionales_PO2_alto_mmHg (separados por comas)",
        "field_high_po2_relative_thresholds_percent": "Umbrales_PO2_alto_relativos_porcentaje_entrada (separados por comas)",
        "field_relative_po2_reference": "Referencia_PO2_alto_relativo (entrada o max_tejido)",
        "result_p50_eff": "P50_eff (mmHg)",
        "result_p_venous": "P_venous (mmHg)",
        "result_p_tissue_min": "P_tissue_min (mmHg)",
        "result_p_tissue_p05": "P_tissue_P05 (mmHg)",
        "result_hypoxic_fraction_lt1": "Fraccion_hipoxica_LT1 (%)",
        "result_hypoxic_fraction_lt5": "Fraccion_hipoxica_LT5 (%)",
        "result_hypoxic_fraction_lt10": "Fraccion_hipoxica_LT10 (%)",
        "result_po2_fraction_gt_primary": "Fraccion_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_secondary": "Fraccion_PO2_GT{threshold} (%)",
        "result_po2_fraction_gt_rel_primary": "Fraccion_PO2_GT{threshold}%_de_{reference} (%)",
        "result_po2_fraction_gt_rel_secondary": "Fraccion_PO2_GT{threshold}%_de_{reference} (%)",
        "result_po2_fraction_gt_rel_tertiary": "Fraccion_PO2_GT{threshold}%_de_{reference} (%)",
        "reference_inlet": "entrada",
        "reference_tissue_max": "max_tejido",
        "result_po2_sensor_avg": "PO2_sensor_avg (mmHg)",
        "result_sa": "S_a (%)",
        "result_sv": "S_v (%)",
        "result_q_flow": "Q_flow_local (nL/s)",
        "output_help_title": "Ayuda de parametros de salida",
        "output_help_intro": "Significado e interpretacion de las salidas actuales del modelo",
        "input_error_title": "Error de entrada",
        "input_error_numeric": "Introduzca valores numericos validos.",
        "input_error_perf": "El factor de perfusion debe ser > 0.",
        "input_error_sweep_param": "Seleccione un parametro de barrido valido.",
        "input_error_sweep_numeric": "Introduzca valores numericos validos para el barrido.",
        "input_error_step_size": "El paso debe ser > 0.",
        "input_error_perf_sweep": "Los valores de barrido del factor de perfusion deben ser > 0.",
        "input_error_select_result": "Seleccione al menos un resultado para el grafico de serie.",
        "info_3d_title": "Grafico 3D",
        "info_3d_mode": "Cambie primero al modo \"Analisis de caso unico\".",
        "save_series_results": "Guardar paquete de ejecucion despues de ver graficos (Excel + graficos + parametros)",
        "save_series_title": "Guardar resultados de la serie como...",
        "save_case_title": "Guardar caso como...",
        "load_case_title": "Cargar caso...",
        "save_error_title": "Error al guardar",
        "load_error_title": "Error al cargar",
        "json_files": "Archivos JSON",
        "all_files": "Todos los archivos",
        "excel_files": "Archivos Excel",
        "default_not_found": "ERROR: no se encontro krogh_basis.py.",
        "running_default": "Ejecutando el script predeterminado...\n",
        "return_code": "Codigo de retorno: {code}",
        "stdout_last": "--- STDOUT (ultimas lineas) ---",
        "stderr": "--- STDERR ---",
        "default_run_error": "ERROR al ejecutar el script predeterminado: {error}",
        "running_single": "Ejecutando la simulacion de caso unico...\n",
        "single_result": "Resultado del caso unico",
        "inputs_header": "  Entradas:",
        "outputs_header": "  Salidas:",
        "single_inputs_line": "    P_inlet={P_inlet:.2f} mmHg, P_MM={P_half:.2f} mmHg, pH={pH:.2f}, pCO2={pCO2:.2f} mmHg, T={temp_c:.2f} C, perf={perf:.2f}, high1={high_po2_threshold_primary:.2f} mmHg, high2={high_po2_threshold_secondary:.2f} mmHg",
        "single_run_error": "ERROR en el analisis de caso unico: {error}",
        "running_series": "Ejecutando el analisis en serie: {parameter} desde {start:.4g} hasta {end:.4g} con paso {step:.4g} ({count} casos)...\n",
        "series_case_progress": "  Caso {index}/{count}: {parameter} = {value:.6g}",
        "series_finished": "Analisis en serie completado.",
        "results_saved": "Resultados guardados en: {path}",
        "plot_saved": "Grafico guardado en: {path}",
        "save_bundle_title": "Seleccione la carpeta para el paquete de ejecucion",
        "bundle_saved": "Paquete de ejecucion guardado en: {path}",
        "bundle_file_parameters": "Parametros de ejecucion guardados en: {path}",
        "bundle_save_cancelled": "Exportacion del paquete omitida.",
        "series_run_error": "ERROR en el analisis en serie: {error}",
        "series_plot_title": "Analisis en serie",
        "publication_mode": "Modo publicacion (etiquetas mas grandes + exportacion alta resolucion)",
        "publication_layout": "Diseno publicacion",
        "publication_layout_a4": "A4 horizontal",
        "publication_layout_wide": "16:9 panoramico",
        "series_plot_sweep": "Barrido: {field} = {start:.4g} a {end:.4g} (n={count})",
        "series_plot_fixed": "Entradas fijas: {params}",
        "series_plot_window": "Grafico de serie",
        "show_figures_title": "Mostrar figuras?",
        "select_figures": "Seleccione las figuras que desea mostrar:",
        "select_all": "Seleccionar todo",
        "deselect_all": "Deseleccionar todo",
        "show_selected": "Mostrar seleccion",
        "cancel": "Cancelar",
        "plot3d_computing": "Calculando grafico 3D...",
        "plot3d_ready": "Grafico 3D listo.",
        "plot3d_error": "ERROR en el grafico 3D: {error}",
        "xlabel_radial_position": "Posicion radial (um)",
        "ylabel_relative_length": "Longitud capilar relativa (x)",
        "zlabel_po2": "PO2 (mmHg)",
        "legend_sensor_avg": "Promedio del sensor (media radial)",
        "title_3d": "Cilindro de Krogh - Caso unico\nP_inlet={P_inlet:.1f} | P_MM={P_half:.2f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f} C | perf={perf:.2f}\nP50_eff={p50_eff:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | PromSensor={sensor_avg:.1f} mmHg",
        "colorbar_po2": "PO2 (mmHg)",
        "case_saved_to": "Caso guardado en: {path}",
        "case_loaded_from": "Caso cargado desde: {path}",
        "sheet_series_results": "Resultados_Serie",
        "sheet_series_setup": "Configuracion_Serie",
        "column_setting": "Configuracion",
        "column_value": "Valor",
        "setting_sweep_parameter": "Parametro variado",
        "setting_start_value": "Valor inicial",
        "setting_end_value": "Valor final",
        "setting_step_size": "Paso",
        "setting_case_count": "Numero de casos",
        "setting_include_axial": "Incluir difusion axial",
        "result_case": "Caso",
        "result_sweep_parameter": "Parametro variado",
        "result_sweep_value": "Valor de barrido",
        "result_include_axial": "Incluir difusion axial",
        "physio_warning_title": "Aviso: rango fisiologico",
        "physio_warning_intro": "Los siguientes valores de entrada estan fuera del rango fisiologicamente plausible. Estos valores solo pueden aparecer en condiciones patologicas extremas o son completamente irrealistas para el tejido microvascular de los mamiferos:\n\n{warnings}\n\nContinuar de todos modos?",
        "physio_warning_low": "  \u2022 {field} = {value:.4g} esta por debajo del minimo plausible de {limit:.4g}",
        "physio_warning_high": "  \u2022 {field} = {value:.4g} esta por encima del maximo plausible de {limit:.4g}",
        "bool_yes": "Si",
        "bool_no": "No",
        "tab_diagnostic": "Diagnostico",
        "run_diagnostic_button": "Ejecutar diagnostico",
        "save_diagnostic_template_button": "Guardar plantilla de calibracion...",
        "reconstruct_krogh_button": "Reconstruir cilindro de Krogh",
        "diag_krogh_computing": "Ajustando modelo de Krogh a los valores diagnosticos...",
        "diag_krogh_ready": "Reconstruccion de Krogh lista.",
        "diag_krogh_error": "ERROR en reconstruccion de Krogh: {error}",
        "diag_krogh_no_result": "Ejecutar primero el diagnostico antes de reconstruir.",
        "title_3d_diagnostic": "Cilindro de Krogh — {state}\nAlerta: {alert} | Riesgo: {risk:.3f} | Confianza: {conf:.3f}\nPO2_inlet={P_inlet:.1f} | pH={pH:.2f} | pCO2={pCO2:.1f} | T={temp_c:.1f}°C\nP_half ajustado={P_half:.2f} mmHg | P_venous={p_venous:.1f} | P_tis_min={p_tis_min:.2f} | PromSensor={sensor_avg:.1f} mmHg",
        "diag_krogh_fit_info": "P_half ajustado={P_half:.3f} mmHg para sensor_po2={sensor_target:.1f} mmHg (simulado={sensor_sim:.1f} mmHg)",
        "diagnostic_group": "Diagnostico probabilistico de oxigenacion (MVP)",
        "diag_po2": "Gases arteriales po2 (mmHg)",
        "diag_pco2": "Gases arteriales pco2 (mmHg)",
        "diag_ph": "pH",
        "diag_temp": "Temperatura (C)",
        "diag_sensor_po2": "Sensor po2 (mmHg)",
        "diag_hemoglobin": "Hemoglobina (g/dL, opcional)",
        "diag_venous_sat": "Saturacion venosa (%, opcional)",
        "diag_optional_hint": "Nota: la hemoglobina y saturacion venosa son opcionales. Dejar en blanco para usar valores por defecto.",
        "diag_yellow_threshold": "Umbral de alerta amarilla (riesgo bajo)",
        "diag_orange_threshold": "Umbral de alerta naranja (riesgo elevado)",
        "diag_red_threshold": "Umbral de alerta roja (critico)",
        "use_single_case_button": "Usar valores del caso unico",
        "diag_result_header": "Resultado del diagnostico",
        "diag_model_missing": "No se pudo cargar el modulo diagnostico oxygenation_diagnostic_mvp.py.",
        "diag_input_error": "Ingresa valores numericos validos para el diagnostico.",
        "diag_threshold_error": "Los umbrales diagnosticos deben cumplir 0 <= amarillo <= naranja <= rojo <= 1.",
        "diag_result_line": "Estado={state} | puntuacion_riesgo={risk_score:.3f} | confianza={confidence:.3f} | certeza={certainty:.3f} | alerta={alert}",
    },
}


def translate(language_code, key, **kwargs):
    language_map = TRANSLATIONS.get(language_code, TRANSLATIONS["en"])
    template = language_map.get(key, TRANSLATIONS["en"].get(key, key))
    return template.format(**kwargs)


def get_numeric_settings():
    return {
        "ode_rtol": float(CAPILLARY_ODE_RTOL),
        "ode_atol": float(CAPILLARY_ODE_ATOL),
        "ode_max_step": float(CAPILLARY_ODE_MAX_STEP),
        "axial_diffusion_max_iter": int(AXIAL_DIFFUSION_MAX_ITER),
        "axial_diffusion_tol": float(AXIAL_DIFFUSION_TOL),
        "axial_coupling_max_iter": int(AXIAL_COUPLING_MAX_ITER),
        "axial_coupling_tol": float(AXIAL_COUPLING_TOL),
    }


def apply_numeric_settings(settings):
    global CAPILLARY_ODE_RTOL, CAPILLARY_ODE_ATOL, CAPILLARY_ODE_MAX_STEP
    global AXIAL_DIFFUSION_MAX_ITER, AXIAL_DIFFUSION_TOL
    global AXIAL_COUPLING_MAX_ITER, AXIAL_COUPLING_TOL

    CAPILLARY_ODE_RTOL = max(float(settings["ode_rtol"]), 1e-14)
    CAPILLARY_ODE_ATOL = max(float(settings["ode_atol"]), 1e-16)
    CAPILLARY_ODE_MAX_STEP = max(float(settings["ode_max_step"]), 1e-8)
    AXIAL_DIFFUSION_MAX_ITER = max(int(settings["axial_diffusion_max_iter"]), 1)
    AXIAL_DIFFUSION_TOL = max(float(settings["axial_diffusion_tol"]), 1e-14)
    AXIAL_COUPLING_MAX_ITER = max(int(settings["axial_coupling_max_iter"]), 1)
    AXIAL_COUPLING_TOL = max(float(settings["axial_coupling_tol"]), 1e-14)


@contextmanager
def temporary_numeric_settings(settings):
    previous_settings = get_numeric_settings()
    apply_numeric_settings(settings)
    try:
        yield
    finally:
        apply_numeric_settings(previous_settings)


def build_tighter_numeric_settings(base_settings):
    return {
        "ode_rtol": max(float(base_settings["ode_rtol"]) * 0.01, 1e-12),
        "ode_atol": max(float(base_settings["ode_atol"]) * 0.01, 1e-14),
        "ode_max_step": max(float(base_settings["ode_max_step"]) * 0.5, 1e-6),
        "axial_diffusion_max_iter": max(int(base_settings["axial_diffusion_max_iter"]) * 2, int(base_settings["axial_diffusion_max_iter"]) + 40),
        "axial_diffusion_tol": max(float(base_settings["axial_diffusion_tol"]) * 0.1, 1e-12),
        "axial_coupling_max_iter": max(int(base_settings["axial_coupling_max_iter"]) * 2, int(base_settings["axial_coupling_max_iter"]) + 4),
        "axial_coupling_tol": max(float(base_settings["axial_coupling_tol"]) * 0.1, 1e-12),
    }


def build_series_check_indices(case_count):
    if case_count <= 25:
        return list(range(case_count))
    return sorted(set(np.linspace(0, case_count - 1, 9, dtype=int).tolist()))


class ToolTip:
    def __init__(self, widget, text_provider, delay_ms=350):
        self.widget = widget
        self.text_provider = text_provider
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_schedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except (RuntimeError, tk.TclError):
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        text = self.text_provider() if callable(self.text_provider) else self.text_provider
        if not text or self._window is not None:
            return
        try:
            self._window = tk.Toplevel(self.widget)
            self._window.wm_overrideredirect(True)
            self._window.attributes("-topmost", True)
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self._window,
                text=text,
                justify="left",
                wraplength=420,
                relief="solid",
                borderwidth=1,
                background="#fffde8",
                padx=8,
                pady=6,
            )
            label.pack()
        except (RuntimeError, tk.TclError):
            self._window = None

    def _hide(self, _event=None):
        self._cancel_schedule()
        if self._window is not None:
            try:
                self._window.destroy()
            except (RuntimeError, tk.TclError):
                pass
            self._window = None

    def destroy(self):
        self._hide()


def effective_p50(pH, pco2, temp_c):
    pco2_safe = max(float(pco2), 1e-6)
    log_shift = (
        BOHR_COEFF * (float(pH) - PH_REF)
        + CO2_COEFF * np.log10(pco2_safe / PCO2_REF)
        + TEMP_COEFF * (float(temp_c) - TEMP_REF)
    )
    return P50 * (10.0 ** log_shift)


def hill_saturation(P, p50_eff=None):
    if p50_eff is None:
        p50_eff = P50
    P_safe = np.maximum(P, 1e-9)
    Pn = P_safe**n_hill
    return Pn / (Pn + p50_eff**n_hill)


def dC_dP(P, p50_eff=None):
    P_safe = np.maximum(P, 1e-9)
    S = hill_saturation(P_safe, p50_eff=p50_eff)
    dS = n_hill * S * (1.0 - S) / P_safe
    return C_Hb * dS + alpha


def krogh_erlang(r, P_c, M, K, R_c, R_t):
    term1 = (M / (4 * K)) * (r**2 - R_c**2)
    term2 = (M * R_t**2 / (2 * K)) * np.log(r / R_c)
    return P_c + term1 - term2


def michaelis_menten_consumption(P_local, P_half, M_max=M_rate):
    P_safe = np.maximum(P_local, 0.0)
    saturation = P_safe / (P_safe + P_half)
    return M_max * (MIN_CONSUMPTION_FRACTION + (1.0 - MIN_CONSUMPTION_FRACTION) * saturation)


def effective_consumption_from_capillary_po2(P_c, P_half, max_iter=30, tol=1e-7):
    P_c_safe = float(max(P_c, 0.0))
    M_eff = M_rate

    for _ in range(max_iter):
        profile = krogh_erlang(r_vec, P_c_safe, M_eff, K_diff, R_cap, R_tis)
        profile = np.maximum(profile, 0.0)
        p_mean = np.average(profile, weights=radial_weights)
        M_new = float(michaelis_menten_consumption(p_mean, P_half=P_half))
        if abs(M_new - M_eff) < tol:
            M_eff = M_new
            break
        M_eff = M_new

    return M_eff


def solve_capillary_profile(dPc_dz, P_inlet):
    sol = solve_ivp(
        dPc_dz,
        (0, L_cap),
        [P_inlet],
        t_eval=z_eval,
        method="RK45",
        rtol=CAPILLARY_ODE_RTOL,
        atol=CAPILLARY_ODE_ATOL,
        max_step=CAPILLARY_ODE_MAX_STEP,
    )
    if not sol.success:
        raise RuntimeError(f"Capillary ODE solver failed: {sol.message}")
    return np.maximum(sol.y[0], 0.0)


def solve_initial_capillary_po2(P_inlet, P_half, p50_eff, perfusion_factor=1.0):
    q_flow_local = max(float(perfusion_factor) * Q_flow, 1e-12)

    def dPc_dz(_, Pc):
        M_eff = effective_consumption_from_capillary_po2(Pc[0], P_half=P_half)
        consumption_per_length = M_eff * np.pi * R_tis**2
        return [-consumption_per_length / (q_flow_local * dC_dP(Pc[0], p50_eff=p50_eff))]

    return solve_capillary_profile(dPc_dz, P_inlet)


def solve_tissue_field_with_axial_diffusion(P_c_axial, P_half, initial_guess=None):
    a_plus = K_diff * (1.0 / dr**2 + 1.0 / (2.0 * r_vec[1:-1] * dr))
    a_minus = K_diff * (1.0 / dr**2 - 1.0 / (2.0 * r_vec[1:-1] * dr))
    a_z = K_diff / dz**2
    denom = 2.0 * K_diff * (1.0 / dr**2 + 1.0 / dz**2)

    if initial_guess is None:
        M_eff_init = np.array(
            [effective_consumption_from_capillary_po2(Pc, P_half=P_half) for Pc in P_c_axial],
            dtype=float,
        )
        P = np.zeros((NZ, NR), dtype=float)
        for i, Pc in enumerate(P_c_axial):
            P[i, :] = krogh_erlang(r_vec, Pc, M_eff_init[i], K_diff, R_cap, R_tis)
    else:
        P = np.array(initial_guess, dtype=float, copy=True)

    P = np.maximum(P, 0.0)
    P[:, 0] = P_c_axial

    for _ in range(AXIAL_DIFFUSION_MAX_ITER):
        P_old = P.copy()
        M_old = michaelis_menten_consumption(P_old[1:-1, 1:-1], P_half=P_half)
        P_new_inner = (
            a_plus[None, :] * P_old[1:-1, 2:]
            + a_minus[None, :] * P_old[1:-1, :-2]
            + a_z * (P_old[2:, 1:-1] + P_old[:-2, 1:-1])
            - M_old
        ) / denom

        P[1:-1, 1:-1] = np.maximum(
            (1.0 - AXIAL_DIFFUSION_RELAX) * P_old[1:-1, 1:-1]
            + AXIAL_DIFFUSION_RELAX * P_new_inner,
            0.0,
        )

        P[:, 0] = P_c_axial
        P[:, -1] = P[:, -2]
        P[0, 1:] = P[1, 1:]
        P[-1, 1:] = P[-2, 1:]
        P[0, 0] = P_c_axial[0]
        P[-1, 0] = P_c_axial[-1]

        if np.max(np.abs(P - P_old)) < AXIAL_DIFFUSION_TOL:
            break

    return np.maximum(P, 0.0)


def solve_axial_capillary_po2(P_inlet, P_half, p50_eff, include_axial_diffusion=True, perfusion_factor=1.0):
    q_flow_local = max(float(perfusion_factor) * Q_flow, 1e-12)
    P_c_axial = solve_initial_capillary_po2(P_inlet, P_half, p50_eff=p50_eff, perfusion_factor=perfusion_factor)
    tissue_po2 = None

    if not include_axial_diffusion:
        M_eff_axial = np.array(
            [effective_consumption_from_capillary_po2(Pc, P_half=P_half) for Pc in P_c_axial],
            dtype=float,
        )
        tissue_po2 = np.zeros((NZ, NR), dtype=float)
        for i, Pc in enumerate(P_c_axial):
            tissue_po2[i, :] = krogh_erlang(r_vec, Pc, M_eff_axial[i], K_diff, R_cap, R_tis)
        return P_c_axial, np.maximum(tissue_po2, 0.0), M_eff_axial

    for _ in range(AXIAL_COUPLING_MAX_ITER):
        tissue_po2 = solve_tissue_field_with_axial_diffusion(P_c_axial, P_half, initial_guess=tissue_po2)
        M_slice = np.average(
            michaelis_menten_consumption(tissue_po2, P_half=P_half),
            axis=1,
            weights=radial_weights,
        )

        def dPc_dz(z_pos, Pc):
            M_eff = np.interp(z_pos, z_eval, M_slice)
            consumption_per_length = M_eff * np.pi * R_tis**2
            return [-consumption_per_length / (q_flow_local * dC_dP(Pc[0], p50_eff=p50_eff))]

        P_new = solve_capillary_profile(dPc_dz, P_inlet)
        if np.max(np.abs(P_new - P_c_axial)) < AXIAL_COUPLING_TOL:
            P_c_axial = P_new
            break
        P_c_axial = P_new

    tissue_po2 = solve_tissue_field_with_axial_diffusion(P_c_axial, P_half, initial_guess=tissue_po2)
    M_eff_axial = np.average(
        michaelis_menten_consumption(tissue_po2, P_half=P_half),
        axis=1,
        weights=radial_weights,
    )
    return P_c_axial, tissue_po2, M_eff_axial


def run_single_case(
    P_inlet,
    P_half,
    pH,
    pCO2,
    temp_c,
    perfusion_factor,
    include_axial_diffusion,
    high_po2_threshold_primary=100.0,
    high_po2_threshold_secondary=200.0,
    additional_high_po2_thresholds=None,
    relative_high_po2_thresholds_percent=None,
    relative_high_po2_reference="inlet",
):
    p50_eff = effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)
    P_c_axial, PO2, _ = solve_axial_capillary_po2(
        P_inlet=P_inlet,
        P_half=P_half,
        p50_eff=p50_eff,
        include_axial_diffusion=include_axial_diffusion,
        perfusion_factor=perfusion_factor,
    )

    tissue_field = np.maximum(PO2, 0.0)
    P_tis_edge = tissue_field[:, -1]
    tissue_values = tissue_field.ravel()
    P_tis_min = float(np.min(tissue_values))
    P_tis_p05 = float(np.percentile(tissue_values, 5.0))
    hypoxic_fraction_lt1 = float(100.0 * np.mean(tissue_values < 1.0))
    hypoxic_fraction_lt5 = float(100.0 * np.mean(tissue_values < 5.0))
    hypoxic_fraction_lt10 = float(100.0 * np.mean(tissue_values < 10.0))
    absolute_thresholds = [float(high_po2_threshold_primary), float(high_po2_threshold_secondary)]
    if additional_high_po2_thresholds:
        absolute_thresholds.extend(float(value) for value in additional_high_po2_thresholds)
    dedup_absolute_thresholds = []
    for threshold in absolute_thresholds:
        if threshold <= 0.0:
            continue
        if any(np.isclose(threshold, existing) for existing in dedup_absolute_thresholds):
            continue
        dedup_absolute_thresholds.append(threshold)
    absolute_fractions = [float(100.0 * np.mean(tissue_values > threshold)) for threshold in dedup_absolute_thresholds]
    abs_fraction_map = dict(zip(dedup_absolute_thresholds, absolute_fractions))
    po2_fraction_gt100 = float(abs_fraction_map.get(float(high_po2_threshold_primary), 0.0))
    po2_fraction_gt200 = float(abs_fraction_map.get(float(high_po2_threshold_secondary), 0.0))

    if relative_high_po2_thresholds_percent is None:
        relative_high_po2_thresholds_percent = [90.0, 50.0, 30.0]
    dedup_relative_thresholds = []
    for percent in relative_high_po2_thresholds_percent:
        percent_value = float(percent)
        if percent_value <= 0.0:
            continue
        if any(np.isclose(percent_value, existing) for existing in dedup_relative_thresholds):
            continue
        dedup_relative_thresholds.append(percent_value)
    if str(relative_high_po2_reference).lower() == "tissue_max":
        relative_reference_value = float(np.max(tissue_values))
        relative_reference_key = "tissue_max"
    else:
        relative_reference_value = float(P_inlet)
        relative_reference_key = "inlet"

    relative_fractions = []
    for percent in dedup_relative_thresholds:
        relative_threshold_mmHg = (percent / 100.0) * relative_reference_value
        relative_fractions.append(float(100.0 * np.mean(tissue_values > relative_threshold_mmHg)))
    rel_fraction_map = dict(zip(dedup_relative_thresholds, relative_fractions))
    rel_values = [
        float(relative_fractions[i]) if i < len(relative_fractions) else float("nan")
        for i in range(3)
    ]
    PO2_avg = np.average(tissue_field, axis=1, weights=radial_weights)
    PO2_avg_axial = float(np.mean(PO2_avg))

    venous = float(P_c_axial[-1])
    s_a = float(100.0 * hill_saturation(P_inlet, p50_eff=p50_eff))
    s_v = float(100.0 * hill_saturation(venous, p50_eff=p50_eff))

    return {
        "P50_eff": float(p50_eff),
        "P_venous": venous,
        "P_tissue_min": P_tis_min,
        "P_tissue_p05": P_tis_p05,
        "Hypoxic_fraction_lt1": hypoxic_fraction_lt1,
        "Hypoxic_fraction_lt5": hypoxic_fraction_lt5,
        "Hypoxic_fraction_lt10": hypoxic_fraction_lt10,
        "PO2_fraction_gt100": po2_fraction_gt100,
        "PO2_fraction_gt200": po2_fraction_gt200,
        "PO2_fraction_gt_rel1": rel_values[0],
        "PO2_fraction_gt_rel2": rel_values[1],
        "PO2_fraction_gt_rel3": rel_values[2],
        "PO2_absolute_thresholds_mmHg": dedup_absolute_thresholds,
        "PO2_fraction_gt_absolute_all": absolute_fractions,
        "PO2_relative_thresholds_percent_of_inlet": dedup_relative_thresholds,
        "PO2_fraction_gt_relative_all": relative_fractions,
        "PO2_relative_reference_key": relative_reference_key,
        "PO2_relative_reference_value_mmHg": relative_reference_value,
        "PO2_sensor_avg": PO2_avg_axial,
        "S_a_percent": s_a,
        "S_v_percent": s_v,
        "Q_flow_nL_s": float(max(perfusion_factor * Q_flow, 1e-12) * 1e6),
    }


def build_series_values(start_value, end_value, step_size):
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


def build_series_case_definitions(
    base_params,
    sweep_field_label,
    sweep_values,
    secondary_field_label=None,
    secondary_values=None,
):
    if sweep_field_label not in SERIES_SWEEP_FIELDS:
        raise ValueError("Unsupported sweep parameter: {}".format(sweep_field_label))
    if secondary_field_label is not None and secondary_field_label not in SERIES_SWEEP_FIELDS:
        raise ValueError("Unsupported secondary sweep parameter: {}".format(secondary_field_label))

    primary_param_name = SERIES_SWEEP_FIELDS[sweep_field_label]
    secondary_param_name = SERIES_SWEEP_FIELDS.get(secondary_field_label)
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


def run_series_cases(base_params, sweep_field_label, start_value, end_value, step_size):
    sweep_values = build_series_values(start_value, end_value, step_size)
    case_definitions = build_series_case_definitions(base_params, sweep_field_label, sweep_values)
    return [build_series_result_row_from_definition(case_definition) for case_definition in case_definitions]


def build_series_result_row(
    case_index,
    sweep_field_label,
    sweep_value,
    case_params,
    secondary_field_label=None,
    secondary_sweep_value=None,
):
    result = run_single_case(
        P_inlet=case_params["P_inlet"],
        P_half=case_params["P_half"],
        pH=case_params["pH"],
        pCO2=case_params["pCO2"],
        temp_c=case_params["temp_c"],
        perfusion_factor=case_params["perf"],
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
        "High_PO2_threshold_1_mmHg": float(case_params["high_po2_threshold_primary"]),
        "High_PO2_threshold_2_mmHg": float(case_params["high_po2_threshold_secondary"]),
        "High_PO2_additional_thresholds_mmHg": ", ".join(f"{value:.6g}" for value in case_params.get("additional_high_po2_thresholds", [])),
        "High_PO2_relative_thresholds_percent": ", ".join(f"{value:.6g}" for value in case_params.get("relative_high_po2_thresholds_percent", [])),
        "Relative_PO2_reference": str(case_params.get("relative_high_po2_reference", "inlet")),
        "Include_axial_diffusion": bool(case_params["include_axial"]),
        **result,
    }


def build_series_result_row_from_definition(case_definition):
    return build_series_result_row(
        case_definition["case_index"],
        case_definition["sweep_field_label"],
        case_definition["sweep_value"],
        case_definition["case_params"],
        secondary_field_label=case_definition.get("secondary_field_label"),
        secondary_sweep_value=case_definition.get("secondary_sweep_value"),
    )


def analyze_series_numerics(case_definitions, results_df, selected_fields, numeric_settings):
    sample_indices = build_series_check_indices(len(case_definitions))
    sampled_case_numbers = [index + 1 for index in sample_indices]
    tighter_settings = build_tighter_numeric_settings(numeric_settings)
    check_rows = []

    with temporary_numeric_settings(tighter_settings):
        for index in sample_indices:
            check_rows.append(build_series_result_row_from_definition(case_definitions[index]))

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
        max_rel_diff = float(rel_diffs[worst_index]) if rel_diffs.size else 0.0
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


class KroghGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self._main_thread_ident = threading.get_ident()
        self.language_code = "en"
        self.language_display_var = tk.StringVar(value=LANGUAGE_NAMES[self.language_code])
        self.title(self.t("app_title"))
        self.geometry("1080x860")
        self.minsize(980, 720)
        self._is_closing = False
        self.protocol("WM_DELETE_WINDOW", self._quit_application)

        self.mode_var = tk.StringVar(value="default")
        self.include_axial_var = tk.BooleanVar(value=True)
        self.save_series_results_var = tk.BooleanVar(value=False)
        self.lock_hypoxic_fraction_scale_var = tk.BooleanVar(value=True)
        self.publication_mode_var = tk.BooleanVar(value=False)
        self.publication_layout_var = tk.StringVar()
        self.publication_layout_key = "wide"
        self.series_dimension_var = tk.StringVar(value="1d")
        self.series_plot_mode_var = tk.StringVar(value="2d")
        self.series_param_var = tk.StringVar()
        self.series_param_key = "PO2_inlet_mmHg"
        self.series_param2_var = tk.StringVar()
        self.series_param2_key = "Perfusion_factor"

        self._build_ui()

    def t(self, key, **kwargs):
        return translate(self.language_code, key, **kwargs)

    def _field_label(self, field_name):
        return self.t(INPUT_FIELD_LABELS.get(field_name, field_name))

    def _result_label(self, field_name):
        return self._result_label_for_context(field_name, None)

    def _result_label_for_context(self, field_name, context):
        if context is None:
            absolute_thresholds = self._get_high_po2_thresholds()
            relative_thresholds = self._get_relative_high_po2_thresholds()
            relative_reference = self._get_relative_reference_mode()
        else:
            absolute_thresholds = context["absolute_thresholds"]
            relative_thresholds = context["relative_thresholds"]
            relative_reference = context["relative_reference"]
        if field_name == "PO2_fraction_gt100":
            threshold = absolute_thresholds[0]
            return self.t("result_po2_fraction_gt_primary", threshold=f"{threshold:.6g}")
        if field_name == "PO2_fraction_gt200":
            threshold = absolute_thresholds[1]
            return self.t("result_po2_fraction_gt_secondary", threshold=f"{threshold:.6g}")
        if field_name == "PO2_fraction_gt_rel1":
            threshold = relative_thresholds[0]
            reference = self.t(f"reference_{relative_reference}")
            return self.t("result_po2_fraction_gt_rel_primary", threshold=f"{threshold:.6g}", reference=reference)
        if field_name == "PO2_fraction_gt_rel2":
            threshold = relative_thresholds[1]
            reference = self.t(f"reference_{relative_reference}")
            return self.t("result_po2_fraction_gt_rel_secondary", threshold=f"{threshold:.6g}", reference=reference)
        if field_name == "PO2_fraction_gt_rel3":
            threshold = relative_thresholds[2]
            reference = self.t(f"reference_{relative_reference}")
            return self.t("result_po2_fraction_gt_rel_tertiary", threshold=f"{threshold:.6g}", reference=reference)
        return self.t(SERIES_RESULT_FIELDS.get(field_name, field_name))

    def _build_result_label_context(self, params):
        relative_thresholds = list(params.get("relative_high_po2_thresholds_percent", [90.0, 50.0, 30.0]))
        while len(relative_thresholds) < 3:
            relative_thresholds.append(relative_thresholds[-1] if relative_thresholds else 30.0)
        return {
            "absolute_thresholds": (
                float(params.get("high_po2_threshold_primary", 100.0)),
                float(params.get("high_po2_threshold_secondary", 200.0)),
            ),
            "relative_thresholds": tuple(float(value) for value in relative_thresholds[:3]),
            "relative_reference": str(params.get("relative_high_po2_reference", "inlet")),
        }

    def _parse_threshold_list(self, raw_text, minimum=0.0, maximum=None):
        cleaned = raw_text.replace(";", ",").strip()
        if not cleaned:
            return []
        values = []
        for chunk in cleaned.split(","):
            token = chunk.strip()
            if not token:
                continue
            token = token.replace("%", "")
            value = float(token)
            if value <= minimum:
                raise ValueError
            if maximum is not None and value > maximum:
                raise ValueError
            values.append(value)
        unique_values = []
        for value in values:
            if any(np.isclose(value, existing) for existing in unique_values):
                continue
            unique_values.append(value)
        return unique_values

    def _normalize_absolute_thresholds(self, primary, secondary, additional):
        all_values = [float(primary), float(secondary)] + [float(value) for value in additional]
        unique_values = []
        for value in all_values:
            if value <= 0.0:
                continue
            if any(np.isclose(value, existing) for existing in unique_values):
                continue
            unique_values.append(value)
        unique_values.sort()
        if len(unique_values) < 2:
            raise ValueError
        return unique_values[0], unique_values[1], unique_values[2:]

    def _parse_relative_reference_mode(self, raw_value):
        """Parse reference mode from display name (translated) or internal key"""
        token = str(raw_value).strip().lower().replace(" ", "").replace("_", "")
        
        # Direct matches for internal keys
        if token in {"", "inlet"}:
            return "inlet"
        if token in {"tissue_max", "tissuemax", "tissue", "max_tissue", "maxtissue"}:
            return "tissue_max"
        
        # Check translated labels and map back to internal keys
        # Try each internal key and see if its translation matches
        for key in ["inlet", "tissue_max"]:
            translated = self.t(f"reference_{key}").strip().lower().replace(" ", "").replace("_", "")
            if translated == token:
                return key
        
        # For backwards compatibility, also check German variants
        if token in {"gewebemax", "maxgewebe"}:
            return "tissue_max"
        
        raise ValueError

    def _get_high_po2_thresholds(self):
        primary = 100.0
        secondary = 200.0
        if hasattr(self, "entries"):
            try:
                primary = float(self.entries["High_PO2_threshold_1_mmHg"].get())
            except Exception:
                primary = 100.0
            try:
                secondary = float(self.entries["High_PO2_threshold_2_mmHg"].get())
            except Exception:
                secondary = 200.0
        return primary, secondary

    def _get_relative_high_po2_thresholds(self):
        defaults = [90.0, 50.0, 30.0]
        if not hasattr(self, "entries"):
            return defaults
        try:
            parsed = self._parse_threshold_list(
                self.entries["High_PO2_relative_thresholds_percent"].get(),
                minimum=0.0,
                maximum=100.0,
            )
            if parsed:
                defaults = parsed
        except Exception:
            pass
        while len(defaults) < 3:
            defaults.append(defaults[-1] if defaults else 30.0)
        return defaults[:3]

    def _get_relative_reference_mode(self):
        if not hasattr(self, "entries"):
            return "inlet"
        try:
            return self._parse_relative_reference_mode(self.entries["Relative_PO2_reference"].get())
        except Exception:
            return "inlet"

    def _result_description(self, field_name):
        descriptions_de = {
            "P50_eff": "effektiver Hb-P50 nach pH-, pCO2- und Temperaturverschiebung; hoehere Werte bedeuten geringere Hb-Affinitat",
            "P_venous": "kapillarer PO2 am venoesen Ende des axialen Profils",
            "P_tissue_min": "globales Minimum des gesamten Gewebefelds und damit sehr sensitiv fuer kleine anoxische Inseln",
            "P_tissue_p05": "5. Perzentil aller Gewebe-PO2-Werte als robustere Hypoxiekenngroesse",
            "Hypoxic_fraction_lt1": "Anteil des Gewebegitters mit PO2 unter 1 mmHg als Marker extremer Anoxie",
            "Hypoxic_fraction_lt5": "Anteil des Gewebegitters mit PO2 unter 5 mmHg als Marker schwerer kritischer Hypoxie",
            "Hypoxic_fraction_lt10": "Anteil des Gewebegitters mit PO2 unter 10 mmHg als fruehere Warnschwelle fuer O2-Mangel",
            "PO2_fraction_gt100": "Anteil des Gewebegitters mit PO2 ueber dem ersten frei waehlbaren Schwellenwert als Marker hyperoxischer Bereiche",
            "PO2_fraction_gt200": "Anteil des Gewebegitters mit PO2 ueber dem zweiten frei waehlbaren Schwellenwert als Marker ausgepraegter Hyperoxie",
            "PO2_fraction_gt_rel1": "Anteil des Gewebegitters ueber dem ersten relativen Inlet-Schwellenwert",
            "PO2_fraction_gt_rel2": "Anteil des Gewebegitters ueber dem zweiten relativen Inlet-Schwellenwert",
            "PO2_fraction_gt_rel3": "Anteil des Gewebegitters ueber dem dritten relativen Inlet-Schwellenwert",
            "PO2_sensor_avg": "radial volumen-gewichteter und axial gemittelter Gewebe-PO2 als Naeherung eines Sensorsignals",
            "S_a_percent": "arterielle Hb-Saettigung aus Einlass-PO2 und effektivem P50",
            "S_v_percent": "venoese Hb-Saettigung aus Endkapillar-PO2 und effektivem P50",
            "Q_flow_nL_s": "lokaler Kapillarfluss nach Skalierung mit dem Perfusionsfaktor",
        }
        descriptions_en = {
            "P50_eff": "effective hemoglobin P50 after pH, pCO2, and temperature shift; higher values mean lower Hb affinity",
            "P_venous": "capillary PO2 at the venous end of the axial profile",
            "P_tissue_min": "global minimum of the full tissue field and therefore highly sensitive to small anoxic islands",
            "P_tissue_p05": "5th percentile of all tissue PO2 values as a more robust hypoxia metric",
            "Hypoxic_fraction_lt1": "fraction of the tissue grid with PO2 below 1 mmHg as a marker of extreme anoxia",
            "Hypoxic_fraction_lt5": "fraction of the tissue grid with PO2 below 5 mmHg as a marker of severe critical hypoxia",
            "Hypoxic_fraction_lt10": "fraction of the tissue grid with PO2 below 10 mmHg as an earlier warning threshold for oxygen deficit",
            "PO2_fraction_gt100": "fraction of the tissue grid above the first user-defined high-PO2 threshold as a marker of hyperoxic regions",
            "PO2_fraction_gt200": "fraction of the tissue grid above the second user-defined high-PO2 threshold as a marker of marked hyperoxia",
            "PO2_fraction_gt_rel1": "fraction of the tissue grid above the first inlet-relative threshold",
            "PO2_fraction_gt_rel2": "fraction of the tissue grid above the second inlet-relative threshold",
            "PO2_fraction_gt_rel3": "fraction of the tissue grid above the third inlet-relative threshold",
            "PO2_sensor_avg": "radially volume-weighted and axially averaged tissue PO2 as a sensor-like summary value",
            "S_a_percent": "arterial Hb saturation from inlet PO2 and effective P50",
            "S_v_percent": "venous Hb saturation from end-capillary PO2 and effective P50",
            "Q_flow_nL_s": "local capillary flow after scaling with the perfusion factor",
        }
        descriptions = descriptions_de if self.language_code == "de" else descriptions_en
        return descriptions.get(field_name, self._result_label(field_name))

    def _format_numeric_value(self, key, value):
        if NUMERIC_SETTINGS_TYPES[key] is int:
            return str(int(round(float(value))))
        return f"{float(value):.6g}"

    def _get_numeric_spec(self, key):
        return NUMERIC_SETTING_SPECS[key]

    def _numeric_description(self, key):
        spec = self._get_numeric_spec(key)
        return spec["description_de"] if self.language_code == "de" else spec["description_en"]

    def _build_numeric_field_tooltip(self, key):
        spec = self._get_numeric_spec(key)
        current_value = spec["default"]
        if hasattr(self, "numeric_entries") and key in self.numeric_entries:
            raw_value = self.numeric_entries[key].get().strip()
            if raw_value:
                current_value = raw_value
        lines = [
            self._numeric_label(next(label_key for field_key, label_key, _ in NUMERIC_SETTINGS_FIELDS if field_key == key)),
            self._numeric_description(key),
            self.t(
                "numeric_help_range_line",
                min_value=self._format_numeric_value(key, spec["min"]),
                max_value=self._format_numeric_value(key, spec["max"]),
                default_value=self._format_numeric_value(key, spec["default"]),
            ),
            self.t("numeric_help_current_line", current_value=current_value),
        ]
        return "\n".join(lines)

    def _format_series_plot_parameters(self, results_df, sweep_field_label, secondary_field_label=None):
        first_row = results_df.iloc[0]
        primary_values = results_df["Sweep_value"].to_numpy(dtype=float)
        fixed_fields = [
            "PO2_inlet_mmHg",
            "mitoP50_mmHg",
            "pH",
            "pCO2_mmHg",
            "Temp_C",
            "Perfusion_factor",
            "High_PO2_threshold_1_mmHg",
            "High_PO2_threshold_2_mmHg",
            "Relative_PO2_reference",
        ]
        fixed_parts = []
        for field_name in fixed_fields:
            if field_name == sweep_field_label or field_name == secondary_field_label:
                continue
            value = first_row[field_name]
            if field_name == "Relative_PO2_reference":
                fixed_parts.append(f"{self._field_label(field_name)}={self.t(f'reference_{value}')}")
            else:
                fixed_parts.append(f"{self._field_label(field_name)}={float(value):.4g}")
        additional_abs = str(first_row.get("High_PO2_additional_thresholds_mmHg", "")).strip()
        if additional_abs:
            fixed_parts.append(f"{self._field_label('High_PO2_additional_thresholds_mmHg')}={additional_abs}")
        relative_percent = str(first_row.get("High_PO2_relative_thresholds_percent", "")).strip()
        if relative_percent:
            fixed_parts.append(f"{self._field_label('High_PO2_relative_thresholds_percent')}={relative_percent}")
        fixed_parts.append(
            f"{self.t('setting_include_axial')}={self._bool_label(bool(first_row['Include_axial_diffusion']))}"
        )
        sweep_text = self.t(
            "series_plot_sweep",
            field=self._field_label(sweep_field_label),
            start=float(np.min(primary_values)),
            end=float(np.max(primary_values)),
            count=int(len(np.unique(np.round(primary_values, decimals=12)))),
        )
        if secondary_field_label:
            secondary_values = results_df["Sweep_value_2"].dropna().to_numpy(dtype=float)
            if secondary_values.size:
                sweep_text += "\n" + self.t(
                    "series_plot_sweep_secondary",
                    field=self._field_label(secondary_field_label),
                    start=float(np.min(secondary_values)),
                    end=float(np.max(secondary_values)),
                    count=int(len(np.unique(np.round(secondary_values, decimals=12)))),
                )
        fixed_text = self.t("series_plot_fixed", params=", ".join(fixed_parts))
        return sweep_text + "\n" + fixed_text

    def _wrap_plot_annotation(self, text, width=118):
        wrapped_lines = []
        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
        return "\n".join(wrapped_lines)

    def _get_series_plot_style(self, publication_mode=False, publication_layout="wide"):
        if publication_mode:
            if publication_layout == "a4":
                return {
                    "figsize_2d": (11.7, 8.3),
                    "figsize_heatmap": (11.7, 8.3),
                    "figsize_3d": (11.7, 8.6),
                    "annotation_fontsize": 9,
                    "legend_fontsize": 9,
                    "wrap_width": 96,
                    "axes_top": 0.68,
                    "save_dpi": 300,
                }
            return {
                "figsize_2d": (14.2, 8.0),
                "figsize_heatmap": (14.2, 8.0),
                "figsize_3d": (14.2, 8.4),
                "annotation_fontsize": 9,
                "legend_fontsize": 9,
                "wrap_width": 108,
                "axes_top": 0.70,
                "save_dpi": 300,
            }
        return {
            "figsize_2d": (13.0, 8.4),
            "figsize_heatmap": (13.0, 8.2),
            "figsize_3d": (13.0, 8.8),
            "annotation_fontsize": 8,
            "legend_fontsize": 8,
            "wrap_width": 118,
            "axes_top": 0.70,
            "save_dpi": 200,
        }

    def _build_output_parameter_help_text(self):
        if self.language_code == "de":
            lines = [
                self.t("output_help_intro"),
                "",
                "Aktueller Modellstand:",
                "- Nichtnegative PO2-Werte werden auf 0 mmHg begrenzt.",
                "- Der Gewebeverbrauch folgt einer Michaelis-Menten-Kinetik mit basal verbleibendem Restverbrauch von 5 % des Maximalverbrauchs.",
                "- Die axiale Gewebediffusion kann zugeschaltet werden und koppelt Gewebefeld und Kapillarprofil iterativ.",
                "",
                f"{self._result_label('P50_eff')}: effektiver Hb-P50 nach pH-, pCO2- und Temperatur-Shift. Ein hoeherer Wert bedeutet geringere Hb-Affinität.",
                f"{self._result_label('P_venous')}: kapillärer PO2 am venösen Ende des axialen Profils.",
                f"{self._result_label('P_tissue_min')}: globales Minimum des gesamten Gewebefelds. Sehr sensitiv, sättigt bei schwerer Hypoxie aber rasch bei 0.",
                f"{self._result_label('P_tissue_p05')}: 5. Perzentil aller Gewebe-PO2-Werte. Robuster als das absolute Minimum und gut für kritische Hypoxie geeignet.",
                f"{self._result_label('Hypoxic_fraction_lt1')}: Anteil des Gewebegitters mit PO2 < 1 mmHg. Marker extremer anoxischer Areale.",
                f"{self._result_label('Hypoxic_fraction_lt5')}: Anteil des Gewebegitters mit PO2 < 5 mmHg. Marker ausgeprägter kritischer Hypoxie.",
                f"{self._result_label('Hypoxic_fraction_lt10')}: Anteil des Gewebegitters mit PO2 < 10 mmHg. Klinisch weichere Frühwarnschwelle für Versorgungseinbußen.",
                f"{self._result_label('PO2_fraction_gt100')}: Anteil des Gewebegitters mit PO2 ueber dem ersten frei waehlbaren Schwellenwert. Marker hyperoxischer Gewebebereiche.",
                f"{self._result_label('PO2_fraction_gt200')}: Anteil des Gewebegitters mit PO2 ueber dem zweiten frei waehlbaren Schwellenwert. Marker ausgepraegter Hyperoxie.",
                f"{self._result_label('PO2_fraction_gt_rel1')}: Anteil des Gewebegitters ueber dem ersten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{self._result_label('PO2_fraction_gt_rel2')}: Anteil des Gewebegitters ueber dem zweiten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{self._result_label('PO2_fraction_gt_rel3')}: Anteil des Gewebegitters ueber dem dritten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{self._result_label('PO2_sensor_avg')}: radial volumen-gewichteter und axial gemittelter Gewebe-PO2. Entspricht am ehesten einem gemittelten Sensorsignal.",
                f"{self._result_label('S_a_percent')}: arterielle Hb-Sättigung aus Einlass-PO2 und P50_eff.",
                f"{self._result_label('S_v_percent')}: venöse Hb-Sättigung aus Endkapillar-PO2 und P50_eff.",
                f"{self._result_label('Q_flow_nL_s')}: lokaler Kapillarfluss nach Skalierung mit dem Perfusionsfaktor.",
            ]
        else:
            lines = [
                self.t("output_help_intro"),
                "",
                "Current model status:",
                "- Non-negative PO2 values are clipped at 0 mmHg.",
                "- Tissue consumption follows Michaelis-Menten kinetics with a residual basal demand of 5 % of maximal consumption.",
                "- Optional axial tissue diffusion couples the tissue field and capillary profile iteratively.",
                "",
                f"{self._result_label('P50_eff')}: effective hemoglobin P50 after pH, pCO2, and temperature shift.",
                f"{self._result_label('P_venous')}: capillary PO2 at the venous end of the axial profile.",
                f"{self._result_label('P_tissue_min')}: global minimum of the full tissue field. Very sensitive, but it saturates at 0 in severe hypoxia.",
                f"{self._result_label('P_tissue_p05')}: 5th percentile of all tissue PO2 values. More robust than the absolute minimum.",
                f"{self._result_label('Hypoxic_fraction_lt1')}: fraction of the tissue grid with PO2 < 1 mmHg.",
                f"{self._result_label('Hypoxic_fraction_lt5')}: fraction of the tissue grid with PO2 < 5 mmHg.",
                f"{self._result_label('Hypoxic_fraction_lt10')}: fraction of the tissue grid with PO2 < 10 mmHg.",
                f"{self._result_label('PO2_fraction_gt100')}: fraction of the tissue grid above the first user-defined threshold.",
                f"{self._result_label('PO2_fraction_gt200')}: fraction of the tissue grid above the second user-defined threshold.",
                f"{self._result_label('PO2_fraction_gt_rel1')}: fraction of the tissue grid above the first inlet-relative threshold.",
                f"{self._result_label('PO2_fraction_gt_rel2')}: fraction of the tissue grid above the second inlet-relative threshold.",
                f"{self._result_label('PO2_fraction_gt_rel3')}: fraction of the tissue grid above the third inlet-relative threshold.",
                f"{self._result_label('PO2_sensor_avg')}: radially volume-weighted and axially averaged tissue PO2, closest to a mean sensor signal.",
                f"{self._result_label('S_a_percent')}: arterial Hb saturation from inlet PO2 and P50_eff.",
                f"{self._result_label('S_v_percent')}: venous Hb saturation from end-capillary PO2 and P50_eff.",
                f"{self._result_label('Q_flow_nL_s')}: local capillary flow after scaling with the perfusion factor.",
            ]
        return "\n".join(lines)

    def _build_numeric_parameter_help_text(self):
        lines = [self.t("numeric_help_intro"), ""]
        for key, label_key, _ in NUMERIC_SETTINGS_FIELDS:
            spec = self._get_numeric_spec(key)
            lines.extend(
                [
                    self._numeric_label(label_key),
                    self._numeric_description(key),
                    self.t(
                        "numeric_help_range_line",
                        min_value=self._format_numeric_value(key, spec["min"]),
                        max_value=self._format_numeric_value(key, spec["max"]),
                        default_value=self._format_numeric_value(key, spec["default"]),
                    ),
                    self.t(
                        "numeric_help_current_line",
                        current_value=self.numeric_entries[key].get().strip() or self._format_numeric_value(key, spec["default"]),
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _show_output_parameter_help(self):
        dlg = tk.Toplevel(self)
        dlg.title(self.t("output_help_title"))
        dlg.geometry("860x620")

        frame = ttk.Frame(dlg)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        text = tk.Text(frame, wrap="word")
        text.insert("1.0", self._build_output_parameter_help_text())
        text.config(state="disabled")
        text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.config(yscrollcommand=scrollbar.set)

        ttk.Button(dlg, text=self.t("cancel"), command=dlg.destroy).pack(pady=(0, 10))

    def _show_numeric_parameter_help(self):
        dlg = tk.Toplevel(self)
        dlg.title(self.t("numeric_help_title"))
        dlg.geometry("860x620")

        frame = ttk.Frame(dlg)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        text = tk.Text(frame, wrap="word")
        text.insert("1.0", self._build_numeric_parameter_help_text())
        text.config(state="disabled")
        text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.config(yscrollcommand=scrollbar.set)

        ttk.Button(dlg, text=self.t("cancel"), command=dlg.destroy).pack(pady=(0, 10))

    def _numeric_label(self, field_name):
        return self.t(field_name)

    def _bool_label(self, value):
        return self.t("bool_yes") if value else self.t("bool_no")

    def _capture_ui_state(self):
        series_entry_keys = {
            "start_value": self.t("start_value"),
            "end_value": self.t("end_value"),
            "step_size": self.t("step_size"),
            "secondary_start_value": self.t("secondary_start_value"),
            "secondary_end_value": self.t("secondary_end_value"),
            "secondary_step_size": self.t("secondary_step_size"),
        }
        state = {
            "mode": self.mode_var.get(),
            "include_axial": self.include_axial_var.get(),
            "save_series_results": self.save_series_results_var.get(),
            "lock_hypoxic_fraction_scale": self.lock_hypoxic_fraction_scale_var.get(),
            "publication_mode": self.publication_mode_var.get(),
            "publication_layout": self.publication_layout_key,
            "series_dimension": self.series_dimension_var.get(),
            "series_plot_mode": self.series_plot_mode_var.get(),
            "series_param_key": self.series_param_key,
            "series_param2_key": self.series_param2_key,
            "entries": {},
            "series_entries": {},
            "numeric_entries": {},
            "output_text": "",
            "plot_selection": [],
            "diagnostic_entries": {},
            "diagnostic_output_text": "",
        }
        if hasattr(self, "entries"):
            state["entries"] = {name: entry.get() for name, entry in self.entries.items()}
        if hasattr(self, "series_entries"):
            state["series_entries"] = {
                key: self.series_entries[label].get()
                for key, label in series_entry_keys.items()
                if label in self.series_entries
            }
        if hasattr(self, "numeric_entries"):
            state["numeric_entries"] = {
                key: entry.get() for key, entry in self.numeric_entries.items()
            }
        if hasattr(self, "output"):
            state["output_text"] = self.output.get("1.0", "end-1c")
        if hasattr(self, "series_plot_listbox"):
            state["plot_selection"] = list(self.series_plot_listbox.curselection())
        if hasattr(self, "diagnostic_entries"):
            state["diagnostic_entries"] = {
                key: entry.get() for key, entry in self.diagnostic_entries.items()
            }
        if hasattr(self, "diagnostic_output"):
            state["diagnostic_output_text"] = self.diagnostic_output.get("1.0", "end-1c")
        if hasattr(self, "series_param_display_to_key"):
            state["series_param_key"] = self.series_param_display_to_key.get(
                self.series_param_var.get(),
                self.series_param_key,
            )
            state["series_param2_key"] = self.series_param_display_to_key.get(
                self.series_param2_var.get(),
                self.series_param2_key,
            )
        return state

    def _restore_ui_state(self, state):
        series_entry_keys = {
            "start_value": self.t("start_value"),
            "end_value": self.t("end_value"),
            "step_size": self.t("step_size"),
            "secondary_start_value": self.t("secondary_start_value"),
            "secondary_end_value": self.t("secondary_end_value"),
            "secondary_step_size": self.t("secondary_step_size"),
        }
        self.mode_var.set(state["mode"])
        self.include_axial_var.set(state["include_axial"])
        self.save_series_results_var.set(state.get("save_series_results", False))
        self.lock_hypoxic_fraction_scale_var.set(state.get("lock_hypoxic_fraction_scale", True))
        self.publication_mode_var.set(state.get("publication_mode", False))
        self._set_publication_layout_display(state.get("publication_layout", "wide"))
        self.series_dimension_var.set(state.get("series_dimension", "1d"))
        self.series_plot_mode_var.set(state.get("series_plot_mode", "2d"))
        self.series_param_key = state["series_param_key"]
        self.series_param2_key = state.get("series_param2_key", self.series_param2_key)
        self._set_series_param_display(self.series_param_key)
        self._set_series_param2_display(self.series_param2_key)

        for name, value in state["entries"].items():
            if name in self.entries:
                widget = self.entries[name]
                if isinstance(widget, ttk.Combobox):
                    # For Relative_PO2_reference, convert internal key to translated label
                    if name == "Relative_PO2_reference":
                        try:
                            # Try to parse as internal key or old value
                            internal_key = self._parse_relative_reference_mode(value)
                            translated_label = self.t(f"reference_{internal_key}")
                            widget.set(translated_label)
                        except ValueError:
                            # Fallback: try direct set (might be outdated format)
                            widget.set(value)
                    else:
                        widget.set(value)
                else:
                    widget.delete(0, "end")
                    widget.insert(0, value)

        for key, value in state["series_entries"].items():
            label = series_entry_keys.get(key)
            if label in self.series_entries:
                self.series_entries[label].delete(0, "end")
                self.series_entries[label].insert(0, value)

        for key, value in state.get("numeric_entries", {}).items():
            if key in self.numeric_entries:
                self.numeric_entries[key].delete(0, "end")
                self.numeric_entries[key].insert(0, value)

        self.series_plot_listbox.selection_clear(0, "end")
        for index in state["plot_selection"]:
            if 0 <= index < self.series_plot_listbox.size():
                self.series_plot_listbox.selection_set(index)

        if state["output_text"]:
            self.output.insert("1.0", state["output_text"])

        for key, value in state.get("diagnostic_entries", {}).items():
            if key in getattr(self, "diagnostic_entries", {}):
                self.diagnostic_entries[key].delete(0, "end")
                self.diagnostic_entries[key].insert(0, value)

        if state.get("diagnostic_output_text") and hasattr(self, "diagnostic_output"):
            self.diagnostic_output.config(state="normal")
            self.diagnostic_output.delete("1.0", "end")
            self.diagnostic_output.insert("1.0", state["diagnostic_output_text"])
            self.diagnostic_output.config(state="disabled")

        self._toggle_inputs()

    def _on_language_selected(self, _event=None):
        selected_code = self.language_name_to_code.get(self.language_display_var.get(), self.language_code)
        if selected_code == self.language_code:
            return
        state = self._capture_ui_state()
        self.language_code = selected_code
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()
        self._restore_ui_state(state)

    def _set_series_param_display(self, field_key):
        self.series_param_key = field_key
        self.series_param_var.set(self.series_param_key_to_display.get(field_key, field_key))

    def _set_series_param2_display(self, field_key):
        self.series_param2_key = field_key
        self.series_param2_var.set(self.series_param_key_to_display.get(field_key, field_key))

    def _set_publication_layout_display(self, layout_key):
        self.publication_layout_key = layout_key if layout_key in {"a4", "wide"} else "wide"
        if hasattr(self, "publication_layout_key_to_display"):
            self.publication_layout_var.set(
                self.publication_layout_key_to_display.get(self.publication_layout_key, self.publication_layout_key)
            )
        else:
            self.publication_layout_var.set(self.publication_layout_key)

    def _on_publication_layout_selected(self, _event=None):
        if hasattr(self, "publication_layout_display_to_key"):
            self.publication_layout_key = self.publication_layout_display_to_key.get(
                self.publication_layout_var.get(),
                "wide",
            )

    def _toggle_series_dimension_inputs(self):
        is_2d = self.mode_var.get() == "series" and self.series_dimension_var.get() == "2d"
        combo_state = "readonly" if is_2d else "disabled"
        entry_state = "normal" if is_2d else "disabled"
        if hasattr(self, "series_param2_combo"):
            self.series_param2_combo.config(state=combo_state)
        if hasattr(self, "series_plot_mode_combo"):
            self.series_plot_mode_combo.config(state=combo_state)
        if hasattr(self, "publication_layout_combo"):
            layout_state = "readonly" if self.mode_var.get() == "series" and self.publication_mode_var.get() else "disabled"
            self.publication_layout_combo.config(state=layout_state)
        for key in ("secondary_start_value", "secondary_end_value", "secondary_step_size"):
            entry = getattr(self, "series_entries_by_key", {}).get(key)
            if entry is not None:
                entry.config(state=entry_state)

    def _build_ui(self):
        self.title(self.t("app_title"))

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        language_frame = ttk.Frame(main_frame)
        language_frame.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(language_frame, text=self.t("language_label")).pack(side="left", padx=(0, 8))
        self.language_name_to_code = {name: code for code, name in LANGUAGE_NAMES.items()}
        self.language_code_to_name = {code: name for code, name in LANGUAGE_NAMES.items()}
        self.language_combo = ttk.Combobox(
            language_frame,
            textvariable=self.language_display_var,
            values=[self.language_code_to_name[code] for code in LANGUAGE_NAMES],
            state="readonly",
            width=14,
        )
        self.language_combo.pack(side="left")
        self.language_display_var.set(self.language_code_to_name[self.language_code])
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)

        top = ttk.LabelFrame(main_frame, text=self.t("mode_group"))
        top.pack(fill="x", padx=10, pady=8)

        ttk.Radiobutton(
            top,
            text=self.t("mode_default"),
            variable=self.mode_var,
            value="default",
            command=self._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Radiobutton(
            top,
            text=self.t("mode_single"),
            variable=self.mode_var,
            value="single",
            command=self._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Radiobutton(
            top,
            text=self.t("mode_series"),
            variable=self.mode_var,
            value="series",
            command=self._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

        self.settings_notebook = ttk.Notebook(main_frame)
        self.settings_notebook.pack(fill="x", padx=10, pady=(0, 8))

        params = ttk.Frame(self.settings_notebook)
        self.params_tab = params
        self.settings_notebook.add(params, text=self.t("tab_inputs"))

        self.entries = {}
        fields = [
            ("PO2_inlet_mmHg", "100.0"),
            ("mitoP50_mmHg", "1.0"),
            ("pH", "7.40"),
            ("pCO2_mmHg", "40.0"),
            ("Temp_C", "37.0"),
            ("Perfusion_factor", "1.0"),
            ("High_PO2_threshold_1_mmHg", "100.0"),
            ("High_PO2_threshold_2_mmHg", "200.0"),
            ("High_PO2_additional_thresholds_mmHg", ""),
            ("High_PO2_relative_thresholds_percent", "90,50,30"),
            ("Relative_PO2_reference", "inlet"),
        ]

        for i, (name, default) in enumerate(fields):
            r = i // 3
            c = (i % 3) * 2
            ttk.Label(params, text=self._field_label(name)).grid(row=r, column=c, padx=8, pady=6, sticky="e")
            if name == "Relative_PO2_reference":
                # Generate combobox values from translations
                ref_values = ("inlet", "tissue_max")
                ref_labels = tuple(self.t(f"reference_{v}") for v in ref_values)
                entry = ttk.Combobox(
                    params,
                    values=ref_labels,
                    state="readonly",
                    width=16,
                )
                # Set default with translated label
                default_label = self.t(f"reference_{default}")
                entry.set(default_label)
            else:
                entry_width = 28 if name in {"High_PO2_additional_thresholds_mmHg", "High_PO2_relative_thresholds_percent"} else 12
                entry = ttk.Entry(params, width=entry_width)
                entry.insert(0, default)
            entry.grid(row=r, column=c + 1, padx=8, pady=6, sticky="w")
            self.entries[name] = entry

        ttk.Checkbutton(
            params,
            text=self.t("include_axial"),
            variable=self.include_axial_var,
        ).grid(row=4, column=0, columnspan=3, padx=8, pady=6, sticky="w")

        series_frame = ttk.Frame(self.settings_notebook)
        self.series_tab = series_frame
        self.settings_notebook.add(series_frame, text=self.t("tab_series"))

        ttk.Label(series_frame, text=self.t("varying_parameter")).grid(row=0, column=0, padx=8, pady=6, sticky="e")
        ttk.Label(series_frame, text=self.t("series_dimension")).grid(row=0, column=2, padx=8, pady=6, sticky="e")
        self.series_param_key_to_display = {
            field_key: self._field_label(field_key) for field_key in SERIES_SWEEP_FIELDS
        }
        self.series_param_display_to_key = {
            label: key for key, label in self.series_param_key_to_display.items()
        }
        self.series_param_combo = ttk.Combobox(
            series_frame,
            textvariable=self.series_param_var,
            values=list(self.series_param_key_to_display.values()),
            state="readonly",
            width=18,
        )
        self.series_param_combo.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        self._set_series_param_display(self.series_param_key)
        self.series_dimension_combo = ttk.Combobox(
            series_frame,
            textvariable=self.series_dimension_var,
            values=("1d", "2d"),
            state="readonly",
            width=10,
        )
        self.series_dimension_combo.grid(row=0, column=3, padx=8, pady=6, sticky="w")
        self.series_dimension_combo.bind("<<ComboboxSelected>>", lambda _event: self._toggle_series_dimension_inputs())

        self.series_entries = {}
        self.series_entries_by_key = {}
        series_fields = [
            ("start_value", self.t("start_value"), "80.0"),
            ("end_value", self.t("end_value"), "120.0"),
            ("step_size", self.t("step_size"), "5.0"),
            ("secondary_start_value", self.t("secondary_start_value"), "0.5"),
            ("secondary_end_value", self.t("secondary_end_value"), "1.5"),
            ("secondary_step_size", self.t("secondary_step_size"), "0.25"),
        ]
        for i, (entry_key, name, default) in enumerate(series_fields):
            row = 1 + i // 3
            column = (i % 3) * 2
            ttk.Label(series_frame, text=name).grid(row=row, column=column, padx=8, pady=6, sticky="e")
            entry_width = 16 if entry_key in {"start_value", "secondary_start_value"} else 12
            entry = ttk.Entry(series_frame, width=entry_width)
            entry.insert(0, default)
            entry.grid(row=row, column=column + 1, padx=8, pady=6, sticky="w")
            self.series_entries[name] = entry
            self.series_entries_by_key[entry_key] = entry

        ttk.Label(series_frame, text=self.t("secondary_parameter")).grid(row=3, column=0, padx=8, pady=6, sticky="e")
        self.series_param2_combo = ttk.Combobox(
            series_frame,
            textvariable=self.series_param2_var,
            values=list(self.series_param_key_to_display.values()),
            state="readonly",
            width=18,
        )
        self.series_param2_combo.grid(row=3, column=1, padx=8, pady=6, sticky="w")
        self._set_series_param2_display(self.series_param2_key)

        ttk.Label(series_frame, text=self.t("series_plot_mode")).grid(row=3, column=2, padx=8, pady=6, sticky="e")
        self.series_plot_mode_combo = ttk.Combobox(
            series_frame,
            textvariable=self.series_plot_mode_var,
            values=("2d", "3d", "heatmap"),
            state="readonly",
            width=18,
        )
        self.series_plot_mode_combo.grid(row=3, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(series_frame, text=self.t("plot_outputs")).grid(row=4, column=0, padx=8, pady=(8, 6), sticky="ne")
        plot_list_frame = ttk.Frame(series_frame)
        plot_list_frame.grid(row=4, column=1, columnspan=5, padx=8, pady=(8, 6), sticky="we")

        self.series_plot_listbox = tk.Listbox(
            plot_list_frame,
            selectmode="extended",
            exportselection=False,
            height=6,
            width=32,
        )
        for field_name in SERIES_RESULT_FIELDS:
            self.series_plot_listbox.insert("end", self._result_label(field_name))
        for index in (1, 2, 3, 4, 6):
            self.series_plot_listbox.selection_set(index)
        self.series_plot_listbox.pack(side="left", fill="x", expand=True)

        plot_scrollbar = ttk.Scrollbar(plot_list_frame, orient="vertical", command=self.series_plot_listbox.yview)
        plot_scrollbar.pack(side="right", fill="y")
        self.series_plot_listbox.config(yscrollcommand=plot_scrollbar.set)

        ttk.Label(
            series_frame,
            text=self.t("multi_select_hint"),
        ).grid(row=5, column=1, columnspan=3, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(
            series_frame,
            text=self.t("series_selection_hint"),
        ).grid(row=6, column=1, columnspan=4, padx=8, pady=(0, 4), sticky="w")
        self.series_save_results_checkbutton = ttk.Checkbutton(
            series_frame,
            text=self.t("save_series_results"),
            variable=self.save_series_results_var,
        )
        self.series_save_results_checkbutton.grid(row=7, column=0, columnspan=3, padx=8, pady=(2, 6), sticky="w")
        self.series_lock_hypoxic_scale_checkbutton = ttk.Checkbutton(
            series_frame,
            text=self.t("lock_hypoxic_fraction_scale"),
            variable=self.lock_hypoxic_fraction_scale_var,
        )
        self.series_lock_hypoxic_scale_checkbutton.grid(row=8, column=0, columnspan=4, padx=8, pady=(0, 6), sticky="w")
        self.series_publication_mode_checkbutton = ttk.Checkbutton(
            series_frame,
            text=self.t("publication_mode"),
            variable=self.publication_mode_var,
            command=self._toggle_series_dimension_inputs,
        )
        self.series_publication_mode_checkbutton.grid(row=9, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(series_frame, text=self.t("publication_layout")).grid(row=9, column=2, padx=8, pady=(0, 6), sticky="e")
        self.publication_layout_key_to_display = {
            "a4": self.t("publication_layout_a4"),
            "wide": self.t("publication_layout_wide"),
        }
        self.publication_layout_display_to_key = {
            label: key for key, label in self.publication_layout_key_to_display.items()
        }
        self.publication_layout_combo = ttk.Combobox(
            series_frame,
            textvariable=self.publication_layout_var,
            values=list(self.publication_layout_key_to_display.values()),
            state="disabled",
            width=18,
        )
        self.publication_layout_combo.grid(row=9, column=3, padx=8, pady=(0, 6), sticky="w")
        self.publication_layout_combo.bind("<<ComboboxSelected>>", self._on_publication_layout_selected)
        self._set_publication_layout_display(self.publication_layout_key)
        ttk.Label(
            series_frame,
            text=self.t("series_plots_separate_hint"),
            wraplength=600,
        ).grid(row=10, column=0, columnspan=6, padx=8, pady=(0, 6), sticky="w")

        self._toggle_series_dimension_inputs()

        numerics_frame = ttk.Frame(self.settings_notebook)
        self.numerics_tab = numerics_frame
        self.settings_notebook.add(numerics_frame, text=self.t("tab_numerics"))

        self.numeric_entries = {}
        self.numeric_tooltips = []
        numeric_defaults = get_numeric_settings()
        for i, (key, label_key, value_type) in enumerate(NUMERIC_SETTINGS_FIELDS):
            row = i // 4
            col = (i % 4) * 2
            label = ttk.Label(numerics_frame, text=self._numeric_label(label_key))
            label.grid(
                row=row,
                column=col,
                padx=8,
                pady=6,
                sticky="e",
            )
            entry = ttk.Entry(numerics_frame, width=14)
            default_value = numeric_defaults[key]
            if value_type is int:
                entry.insert(0, str(int(default_value)))
            else:
                entry.insert(0, f"{float(default_value):.6g}")
            entry.grid(row=row, column=col + 1, padx=8, pady=6, sticky="w")
            entry.bind("<FocusOut>", lambda _event, setting_key=key: self._sanitize_numeric_entry(setting_key), add="+")
            entry.bind("<Return>", lambda _event, setting_key=key: self._sanitize_numeric_entry(setting_key), add="+")
            self.numeric_entries[key] = entry
            self.numeric_tooltips.append(ToolTip(label, lambda setting_key=key: self._build_numeric_field_tooltip(setting_key)))
            self.numeric_tooltips.append(ToolTip(entry, lambda setting_key=key: self._build_numeric_field_tooltip(setting_key)))

        ttk.Button(
            numerics_frame,
            text=self.t("numeric_help_button"),
            command=self._show_numeric_parameter_help,
        ).grid(row=2, column=0, padx=8, pady=(2, 8), sticky="w")
        ttk.Label(
            numerics_frame,
            text=self.t("numeric_help_hint"),
            wraplength=700,
        ).grid(row=2, column=1, columnspan=7, padx=8, pady=(2, 8), sticky="w")

        diagnostic_frame = ttk.Frame(self.settings_notebook)
        self.diagnostic_tab = diagnostic_frame
        self.settings_notebook.add(diagnostic_frame, text=self.t("tab_diagnostic"))

        diagnostic_group = ttk.LabelFrame(diagnostic_frame, text=self.t("diagnostic_group"))
        diagnostic_group.pack(fill="x", padx=10, pady=10)

        self.diagnostic_entries = {}
        diagnostic_fields = [
            ("po2", "diag_po2", "80.0"),
            ("pco2", "diag_pco2", "40.0"),
            ("pH", "diag_ph", "7.40"),
            ("temperature_c", "diag_temp", "37.0"),
            ("sensor_po2", "diag_sensor_po2", "65.0"),
            ("hemoglobin_g_dl", "diag_hemoglobin", ""),
            ("venous_sat_percent", "diag_venous_sat", ""),
            ("yellow_threshold", "diag_yellow_threshold", "0.40"),
            ("orange_threshold", "diag_orange_threshold", "0.60"),
            ("red_threshold", "diag_red_threshold", "0.80"),
        ]
        for i, (field_key, label_key, default_value) in enumerate(diagnostic_fields):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(diagnostic_group, text=self.t(label_key)).grid(row=row, column=col, padx=8, pady=6, sticky="e")
            entry = ttk.Entry(diagnostic_group, width=12)
            entry.insert(0, default_value)
            entry.grid(row=row, column=col + 1, padx=8, pady=6, sticky="w")
            self.diagnostic_entries[field_key] = entry

        diagnostic_controls = ttk.Frame(diagnostic_group)
        diagnostic_controls.grid(row=4, column=0, columnspan=6, padx=8, pady=(4, 8), sticky="w")
        ttk.Label(diagnostic_group, text=self.t("diag_optional_hint")).grid(row=3, column=0, columnspan=6, padx=8, pady=(0, 4), sticky="w")
        self.use_single_case_button = ttk.Button(
            diagnostic_controls,
            text=self.t("use_single_case_button"),
            command=self._fill_diagnostic_from_single_case,
        )
        self.use_single_case_button.pack(side="left", padx=(0, 6))
        self.run_diagnostic_button = ttk.Button(
            diagnostic_controls,
            text=self.t("run_diagnostic_button"),
            command=self._run_diagnostic_from_inputs,
        )
        self.run_diagnostic_button.pack(side="left", padx=(0, 6))
        self.save_diagnostic_template_button = ttk.Button(
            diagnostic_controls,
            text=self.t("save_diagnostic_template_button"),
            command=self._save_diagnostic_calibration_template,
        )
        self.save_diagnostic_template_button.pack(side="left")

        self.reconstruct_krogh_button = ttk.Button(
            diagnostic_controls,
            text=self.t("reconstruct_krogh_button"),
            command=self._run_reconstruct_krogh,
        )
        self.reconstruct_krogh_button.pack(side="left", padx=(6, 0))

        self.diagnostic_output = tk.Text(diagnostic_frame, height=8, wrap="word")
        self.diagnostic_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.diagnostic_output.insert("1.0", self.t("diag_result_header") + "\n")
        self.diagnostic_output.config(state="disabled")

        controls = ttk.Frame(main_frame)
        controls.pack(fill="x", padx=10, pady=8)

        ttk.Button(controls, text=self.t("run_button"), command=self._run).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("plot3d_button"), command=self._run_3d_plot).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("clear_button"), command=self._clear).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("help_button"), command=self._show_output_parameter_help).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("save_case_button"), command=self._save_case).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("load_case_button"), command=self._load_case).pack(side="left", padx=4)
        ttk.Button(controls, text=self.t("quit_button"), command=self._quit_application).pack(side="right", padx=4)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", padx=10, pady=(0, 4))

        self.progress = ttk.Progressbar(progress_frame, mode="indeterminate", length=400)
        self.progress.pack(fill="x", pady=(0, 4))

        self.status_var = tk.StringVar(value=self.t("status_ready"))
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")

        out_frame = ttk.LabelFrame(main_frame, text=self.t("output_group"))
        out_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        output_inner = ttk.Frame(out_frame)
        output_inner.pack(fill="both", expand=True, padx=6, pady=6)

        self.output = tk.Text(output_inner, wrap="word")
        self.output.pack(side="left", fill="both", expand=True)

        output_scrollbar = ttk.Scrollbar(output_inner, orient="vertical", command=self.output.yview)
        output_scrollbar.pack(side="right", fill="y")
        self.output.config(yscrollcommand=output_scrollbar.set)

        self._toggle_inputs()

    def _call_on_ui_thread(self, callback, *args, **kwargs):
        if self._is_closing:
            return
        try:
            if threading.get_ident() == self._main_thread_ident:
                callback(*args, **kwargs)
            else:
                self.after(0, lambda: (not self._is_closing) and callback(*args, **kwargs))
        except (RuntimeError, tk.TclError):
            pass

    def _append(self, text):
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _append_async(self, text):
        self._call_on_ui_thread(self._append, text)

    def _set_status(self, text):
        if hasattr(self, "status_var"):
            self.status_var.set(text)

    def _set_status_async(self, text):
        self._call_on_ui_thread(self._set_status, text)

    def _set_progress_running(self, running):
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _check_physiological_warnings(self, checks):
        """Check a list of (field_key, value) pairs against PHYSIOLOGICAL_RANGES.
        Shows a single askyesno dialog if any violations are found.
        Returns True if validation should proceed, False if the user cancels.
        """
        violations = []
        for field_key, value in checks:
            bounds = PHYSIOLOGICAL_RANGES.get(field_key)
            if bounds is None:
                continue
            label = self._field_label(field_key)
            if value < bounds["warn_low"]:
                violations.append(
                    self.t("physio_warning_low", field=label, value=value, limit=bounds["warn_low"])
                )
            elif value > bounds["warn_high"]:
                violations.append(
                    self.t("physio_warning_high", field=label, value=value, limit=bounds["warn_high"])
                )
        if not violations:
            return True
        warning_text = self.t(
            "physio_warning_intro",
            warnings="\n".join(violations),
        )
        return messagebox.askyesno(self.t("physio_warning_title"), warning_text)

    def _get_single_case_inputs(self):
        try:
            params = {
                "P_inlet": float(self.entries["PO2_inlet_mmHg"].get()),
                "P_half": float(self.entries["mitoP50_mmHg"].get()),
                "pH": float(self.entries["pH"].get()),
                "pCO2": float(self.entries["pCO2_mmHg"].get()),
                "temp_c": float(self.entries["Temp_C"].get()),
                "perf": float(self.entries["Perfusion_factor"].get()),
                "high_po2_threshold_primary": float(self.entries["High_PO2_threshold_1_mmHg"].get()),
                "high_po2_threshold_secondary": float(self.entries["High_PO2_threshold_2_mmHg"].get()),
                "additional_high_po2_thresholds": self._parse_threshold_list(
                    self.entries["High_PO2_additional_thresholds_mmHg"].get(),
                    minimum=0.0,
                ),
                "relative_high_po2_thresholds_percent": self._parse_threshold_list(
                    self.entries["High_PO2_relative_thresholds_percent"].get(),
                    minimum=0.0,
                    maximum=100.0,
                ),
                "relative_high_po2_reference": self._parse_relative_reference_mode(
                    self.entries["Relative_PO2_reference"].get()
                ),
                "include_axial": self.include_axial_var.get(),
            }
        except ValueError:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_numeric"))
            return None

        if params["perf"] <= 0.0:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_perf"))
            return None

        if params["high_po2_threshold_primary"] <= 0.0 or params["high_po2_threshold_secondary"] <= 0.0:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_numeric"))
            return None
        if not params["relative_high_po2_thresholds_percent"]:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_numeric"))
            return None

        try:
            p_sorted, s_sorted, add_sorted = self._normalize_absolute_thresholds(
                params["high_po2_threshold_primary"],
                params["high_po2_threshold_secondary"],
                params["additional_high_po2_thresholds"],
            )
        except ValueError:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_numeric"))
            return None

        params["high_po2_threshold_primary"] = p_sorted
        params["high_po2_threshold_secondary"] = s_sorted
        params["additional_high_po2_thresholds"] = add_sorted
        self.entries["High_PO2_threshold_1_mmHg"].delete(0, "end")
        self.entries["High_PO2_threshold_1_mmHg"].insert(0, f"{p_sorted:.6g}")
        self.entries["High_PO2_threshold_2_mmHg"].delete(0, "end")
        self.entries["High_PO2_threshold_2_mmHg"].insert(0, f"{s_sorted:.6g}")
        self.entries["High_PO2_additional_thresholds_mmHg"].delete(0, "end")
        self.entries["High_PO2_additional_thresholds_mmHg"].insert(
            0,
            ",".join(f"{value:.6g}" for value in add_sorted),
        )
        # For Combobox, use .set() with translated label instead of .insert()
        self.entries["Relative_PO2_reference"].set(
            self.t(f"reference_{params['relative_high_po2_reference']}")
        )

        physio_checks = [
            ("PO2_inlet_mmHg", params["P_inlet"]),
            ("mitoP50_mmHg",   params["P_half"]),
            ("pH",             params["pH"]),
            ("pCO2_mmHg",      params["pCO2"]),
            ("Temp_C",         params["temp_c"]),
            ("Perfusion_factor", params["perf"]),
        ]
        if not self._check_physiological_warnings(physio_checks):
            return None

        return params

    def _get_series_inputs(self):
        base_params = self._get_single_case_inputs()
        if base_params is None:
            return None

        numeric_settings = self._get_numeric_settings_inputs()
        if numeric_settings is None:
            return None

        sweep_field_label = self.series_param_var.get()
        sweep_field_key = self.series_param_display_to_key.get(sweep_field_label)
        if sweep_field_key not in SERIES_SWEEP_FIELDS:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_sweep_param"))
            return None

        is_2d = self.series_dimension_var.get() == "2d"
        secondary_field_key = None
        if is_2d:
            secondary_field_label = self.series_param2_var.get()
            secondary_field_key = self.series_param_display_to_key.get(secondary_field_label)
            if secondary_field_key not in SERIES_SWEEP_FIELDS:
                messagebox.showerror(self.t("input_error_title"), self.t("input_error_sweep_param"))
                return None
            if secondary_field_key == sweep_field_key:
                messagebox.showerror(self.t("input_error_title"), self.t("input_error_sweep_param_duplicate"))
                return None

        try:
            start_value = float(self.series_entries_by_key["start_value"].get())
            end_value = float(self.series_entries_by_key["end_value"].get())
            step_size = float(self.series_entries_by_key["step_size"].get())
        except ValueError:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_sweep_numeric"))
            return None

        secondary_start_value = None
        secondary_end_value = None
        secondary_step_size = None
        if is_2d:
            try:
                secondary_start_value = float(self.series_entries_by_key["secondary_start_value"].get())
                secondary_end_value = float(self.series_entries_by_key["secondary_end_value"].get())
                secondary_step_size = float(self.series_entries_by_key["secondary_step_size"].get())
            except ValueError:
                messagebox.showerror(self.t("input_error_title"), self.t("input_error_secondary_sweep_numeric"))
                return None

        if step_size <= 0.0:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_step_size"))
            return None

        if is_2d and secondary_step_size <= 0.0:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_secondary_step_size"))
            return None

        if sweep_field_key == "Perfusion_factor" and (start_value <= 0.0 or end_value <= 0.0):
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_perf_sweep"))
            return None

        if secondary_field_key == "Perfusion_factor" and (secondary_start_value <= 0.0 or secondary_end_value <= 0.0):
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_secondary_perf_sweep"))
            return None

        selected_indices = self.series_plot_listbox.curselection()
        if not selected_indices:
            messagebox.showerror(self.t("input_error_title"), self.t("input_error_select_result"))
            return None

        result_keys = list(SERIES_RESULT_FIELDS.keys())
        selected_plot_fields = [result_keys[index] for index in selected_indices]
        publication_layout = self.publication_layout_display_to_key.get(
            self.publication_layout_var.get(),
            self.publication_layout_key,
        )
        self.publication_layout_key = publication_layout

        # Check physiological range for the sweep extent (base params already
        # checked inside _get_single_case_inputs above).
        sweep_physio_checks = [
            (sweep_field_key, start_value),
            (sweep_field_key, end_value),
        ]
        if secondary_field_key is not None:
            sweep_physio_checks += [
                (secondary_field_key, secondary_start_value),
                (secondary_field_key, secondary_end_value),
            ]
        if not self._check_physiological_warnings(sweep_physio_checks):
            return None

        return {
            "base_params": base_params,
            "numeric_settings": numeric_settings,
            "save_bundle_after_display": self.save_series_results_var.get(),
            "lock_hypoxic_fraction_scale": self.lock_hypoxic_fraction_scale_var.get(),
            "publication_mode": self.publication_mode_var.get(),
            "publication_layout": publication_layout,
            "series_dimension": self.series_dimension_var.get(),
            "series_plot_mode": self.series_plot_mode_var.get(),
            "sweep_field_label": sweep_field_key,
            "start_value": start_value,
            "end_value": end_value,
            "step_size": step_size,
            "secondary_field_label": secondary_field_key,
            "secondary_start_value": secondary_start_value,
            "secondary_end_value": secondary_end_value,
            "secondary_step_size": secondary_step_size,
            "selected_plot_fields": selected_plot_fields,
            "result_label_context": self._build_result_label_context(base_params),
        }

    def _get_numeric_settings_inputs(self):
        settings = {}
        try:
            for key, _, value_type in NUMERIC_SETTINGS_FIELDS:
                value = self._sanitize_numeric_entry(key)
                spec = self._get_numeric_spec(key)
                if value < spec["min"] or value > spec["max"]:
                    raise ValueError(key)
                settings[key] = value
        except (KeyError, ValueError, TypeError) as exc:
            if isinstance(exc, ValueError) and str(exc) in NUMERIC_SETTING_SPECS:
                key = str(exc)
                spec = self._get_numeric_spec(key)
                label_key = next(label_key for field_key, label_key, _ in NUMERIC_SETTINGS_FIELDS if field_key == key)
                message = self.t(
                    "input_error_numeric_range",
                    field=self._numeric_label(label_key),
                    min_value=self._format_numeric_value(key, spec["min"]),
                    max_value=self._format_numeric_value(key, spec["max"]),
                    default_value=self._format_numeric_value(key, spec["default"]),
                )
            else:
                message = self.t("input_error_numerics")
            messagebox.showerror(self.t("input_error_title"), message)
            return None
        return settings

    def _sanitize_numeric_entry(self, key):
        spec = self._get_numeric_spec(key)
        value_type = NUMERIC_SETTINGS_TYPES[key]
        entry = self.numeric_entries[key]
        raw_value = entry.get().strip()
        try:
            value = value_type(raw_value)
            if not np.isfinite(float(value)):
                raise ValueError
        except (TypeError, ValueError):
            value = value_type(spec["default"])
        value = max(spec["min"], min(spec["max"], value))
        entry.delete(0, "end")
        entry.insert(0, self._format_numeric_value(key, value))
        return value

    def _toggle_inputs(self):
        state = "normal" if self.mode_var.get() in {"single", "series"} else "disabled"
        for name, entry in self.entries.items():
            if name == "Relative_PO2_reference":
                entry.config(state="readonly" if state == "normal" else "disabled")
            else:
                entry.config(state=state)

        series_state = "readonly" if self.mode_var.get() == "series" else "disabled"
        self.series_param_combo.config(state=series_state)
        self.series_dimension_combo.config(state=series_state)
        entry_state = "normal" if self.mode_var.get() == "series" else "disabled"
        for entry in self.series_entries_by_key.values():
            entry.config(state=entry_state)
        self.series_plot_listbox.config(state="normal" if self.mode_var.get() == "series" else "disabled")
        self.series_save_results_checkbutton.config(state="normal" if self.mode_var.get() == "series" else "disabled")
        self.series_lock_hypoxic_scale_checkbutton.config(state="normal" if self.mode_var.get() == "series" else "disabled")
        self.series_publication_mode_checkbutton.config(state="normal" if self.mode_var.get() == "series" else "disabled")
        for entry in self.numeric_entries.values():
            entry.config(state=state)
        for entry in getattr(self, "diagnostic_entries", {}).values():
            entry.config(state=state)
        if hasattr(self, "run_diagnostic_button"):
            self.run_diagnostic_button.config(state=state)
        if hasattr(self, "save_diagnostic_template_button"):
            self.save_diagnostic_template_button.config(state=state)

        if hasattr(self, "settings_notebook"):
            self.settings_notebook.tab(self.series_tab, state="normal" if self.mode_var.get() == "series" else "hidden")
            if self.mode_var.get() == "series":
                self.settings_notebook.select(self.series_tab)
            else:
                self.settings_notebook.select(self.params_tab)

        self._toggle_series_dimension_inputs()

    def _set_diagnostic_output(self, text):
        self.diagnostic_output.config(state="normal")
        self.diagnostic_output.delete("1.0", "end")
        self.diagnostic_output.insert("1.0", text.strip() + "\n")
        self.diagnostic_output.config(state="disabled")

    def _fill_diagnostic_from_single_case(self):
        if not hasattr(self, "entries"):
            messagebox.showerror(self.t("input_error_title"), "Diagnostic inputs not available.")
            return
        try:
            self.diagnostic_entries["po2"].delete(0, "end")
            self.diagnostic_entries["po2"].insert(0, self.entries["PO2_inlet_mmHg"].get())
            self.diagnostic_entries["pco2"].delete(0, "end")
            self.diagnostic_entries["pco2"].insert(0, self.entries["pCO2_mmHg"].get())
            self.diagnostic_entries["pH"].delete(0, "end")
            self.diagnostic_entries["pH"].insert(0, self.entries["pH"].get())
            self.diagnostic_entries["temperature_c"].delete(0, "end")
            self.diagnostic_entries["temperature_c"].insert(0, self.entries["Temp_C"].get())
        except Exception as exc:
            messagebox.showerror(self.t("input_error_title"), f"Could not copy values: {exc}")

    def _format_oxygenation_state_label(self, state_name):
        language_var = self.__dict__.get("language_var") if hasattr(self, "__dict__") else None
        language = language_var.get() if language_var is not None else "English"
        labels = {
            "English": {
                "normoxia": "normoxia",
                "mild_hypoxia": "mild tissue hypoxia",
                "compensated_hypoxia": "compensated tissue hypoxia",
                "severe_hypoxia": "severe tissue hypoxia",
                "profound_hypoxia": "profound tissue hypoxia",
            },
            "Deutsch": {
                "normoxia": "Normoxie",
                "mild_hypoxia": "milde Gewebehypoxie",
                "compensated_hypoxia": "kompensierte Gewebehypoxie",
                "severe_hypoxia": "schwere Gewebehypoxie",
                "profound_hypoxia": "ausgepraegte Gewebehypoxie",
            },
        }
        fallback = labels["English"]
        return labels.get(language, fallback).get(state_name, str(state_name).replace("_", " "))

    def _run_diagnostic_from_inputs(self):
        if OxygenationInput is None or alert_decision is None:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_model_missing"))
            return
        try:
            po2 = float(self.diagnostic_entries["po2"].get())
            pco2 = float(self.diagnostic_entries["pco2"].get())
            ph = float(self.diagnostic_entries["pH"].get())
            temperature_c = float(self.diagnostic_entries["temperature_c"].get())
            sensor_po2 = float(self.diagnostic_entries["sensor_po2"].get())
            hb_str = self.diagnostic_entries["hemoglobin_g_dl"].get().strip()
            hemoglobin = float(hb_str) if hb_str else 13.5
            sv_str = self.diagnostic_entries["venous_sat_percent"].get().strip()
            venous_sat = float(sv_str) if sv_str else 75.0
            yellow_threshold = float(self.diagnostic_entries["yellow_threshold"].get())
            orange_threshold = float(self.diagnostic_entries["orange_threshold"].get())
            red_threshold = float(self.diagnostic_entries["red_threshold"].get())
        except ValueError:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_input_error"))
            return

        if not (0.0 <= yellow_threshold <= orange_threshold <= red_threshold <= 1.0):
            messagebox.showerror(self.t("input_error_title"), 
                               "Thresholds must satisfy 0 ≤ yellow ≤ orange ≤ red ≤ 1.0")
            return

        result = alert_decision(
            OxygenationInput(
                po2=po2,
                pco2=pco2,
                pH=ph,
                temperature_c=temperature_c,
                sensor_po2=sensor_po2,
                hemoglobin_g_dl=hemoglobin,
                venous_sat_percent=venous_sat,
            ),
            yellow_threshold=yellow_threshold,
            orange_threshold=orange_threshold,
            red_threshold=red_threshold,
        )

        output_lines = [self.t("diag_result_header")]
        display_state = self._format_oxygenation_state_label(result['predicted_state'])
        output_lines.append(
            f"State: {display_state} | "
            f"Risk: {result.get('risk_score', 'N/A'):.3f} | "
            f"Alert: {result['alert_level']} | "
            f"Confidence: {float(result['confidence']):.3f} | "
            f"Certainty: {float(result['certainty']):.3f}"
        )
        output_lines.append(f"p_normoxia={float(result['p_normoxia']):.3f}")
        output_lines.append(f"p_mild_tissue_hypoxia={float(result['p_mild_hypoxia']):.3f}")
        output_lines.append(f"p_compensated_tissue_hypoxia={float(result['p_compensated_hypoxia']):.3f}")
        output_lines.append(f"p_severe_tissue_hypoxia={float(result['p_severe_hypoxia']):.3f}")
        output_lines.append(f"p_profound_tissue_hypoxia={float(result['p_profound_hypoxia']):.3f}")
        self._set_diagnostic_output("\n".join(output_lines))
        self._append("[Diagnostic] " + output_lines[1])

    def _save_diagnostic_calibration_template(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[(self.t("all_files"), "*.csv")],
            title=self.t("save_series_title"),
        )
        if not path:
            return
        template = pd.DataFrame(
            [
                {
                    "case_id": "case_001",
                    "po2": float(self.diagnostic_entries["po2"].get() or 80.0),
                    "pco2": float(self.diagnostic_entries["pco2"].get() or 40.0),
                    "pH": float(self.diagnostic_entries["pH"].get() or 7.4),
                    "temperature_c": float(self.diagnostic_entries["temperature_c"].get() or 37.0),
                    "sensor_po2": float(self.diagnostic_entries["sensor_po2"].get() or 65.0),
                    "true_state": "",
                }
            ]
        )
        try:
            template.to_csv(path, index=False)
            self._append(self.t("results_saved", path=path))
        except Exception as exc:
            messagebox.showerror(self.t("save_error_title"), str(exc))

    # ------------------------------------------------------------------
    # Krogh reconstruction from diagnostic result
    # ------------------------------------------------------------------

    def _fit_p_half_from_venous(self, P_inlet, P_v_target, pH, pCO2, temp_c,
                                 perfusion_factor=1.0, include_axial=True):
        """Find mitoP50 (P_half) such that the simulated capillary outlet PO2 (P_venous)
        matches the blood-gas-derived venous PO2 target.

        P_v_target must be computed from venous_sat_percent via the Hill equation before
        calling this method.  Returns (P_half_fitted, P_venous_simulated).
        """
        p50_eff = effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)

        def _simulate_venous(P_half_candidate):
            P_c_axial, _, _ = solve_axial_capillary_po2(
                P_inlet=P_inlet,
                P_half=P_half_candidate,
                p50_eff=p50_eff,
                include_axial_diffusion=include_axial,
                perfusion_factor=perfusion_factor,
            )
            return float(P_c_axial[-1])

        def _residual(P_half_candidate):
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

    def _fit_joint_krogh_parameters(self, P_inlet, sensor_target, P_v_target, pH, pCO2, temp_c,
                                    include_axial=True, venous_weight=0.15):
        """Jointly fit perfusion and mitoP50 so the reconstruction reflects both
        the measured tissue sensor PO2 and the venous saturation target.

        Unlike the earlier approximation, this performs a real 2D search over both
        perfusion and mitoP50. The venous saturation is treated as a soft constraint,
        especially when the optional default value is used.
        """
        p50_eff = effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)
        perf_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.35, 5.0, 15))))
        p_half_candidates = np.unique(np.concatenate(([1.0], np.geomspace(0.05, 300.0, 22))))
        best = None

        for perf in perf_candidates:
            for p_half in p_half_candidates:
                try:
                    P_c_axial, tissue_po2, _ = solve_axial_capillary_po2(
                        P_inlet=P_inlet,
                        P_half=float(p_half),
                        p50_eff=p50_eff,
                        include_axial_diffusion=include_axial,
                        perfusion_factor=float(perf),
                    )
                    tissue_po2 = np.maximum(tissue_po2, 0.0)
                    sensor_sim = float(np.average(tissue_po2, axis=1, weights=radial_weights).mean())
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
                    if best is None or candidate["objective"] < best["objective"]:
                        best = candidate
                except Exception:
                    continue

        if best is None:
            raise RuntimeError("Could not fit Krogh parameters for the selected diagnostic inputs.")

        best["fit_warning"] = bool(best["sensor_error"] > 6.0 or (float(venous_weight) >= 0.4 and best["venous_error"] > 4.0))
        return best

    def _run_reconstruct_krogh(self):
        """Triggered by the 'Reconstruct Krogh Cylinder' button."""
        if OxygenationInput is None or alert_decision is None:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_model_missing"))
            return

        # Re-read current diagnostic inputs
        try:
            po2 = float(self.diagnostic_entries["po2"].get())
            pco2 = float(self.diagnostic_entries["pco2"].get())
            ph = float(self.diagnostic_entries["pH"].get())
            temperature_c = float(self.diagnostic_entries["temperature_c"].get())
            sensor_po2 = float(self.diagnostic_entries["sensor_po2"].get())
            venous_sat_raw = self.diagnostic_entries["venous_sat_percent"].get().strip()
            hemoglobin_raw = self.diagnostic_entries["hemoglobin_g_dl"].get().strip()
            venous_sat = float(venous_sat_raw) if venous_sat_raw else 75.0
            hemoglobin = float(hemoglobin_raw) if hemoglobin_raw else 13.5
            yellow_threshold = float(self.diagnostic_entries["yellow_threshold"].get())
            orange_threshold = float(self.diagnostic_entries["orange_threshold"].get())
            red_threshold = float(self.diagnostic_entries["red_threshold"].get())
        except ValueError:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_input_error"))
            return

        # Optional venous saturation acts as a soft hint. If left blank or left at the
        # common default value, it should not dominate the 3D reconstruction.
        if not venous_sat_raw or abs(venous_sat - 75.0) < 1e-6:
            venous_weight = 0.15
        else:
            venous_weight = 0.55

        # Re-run diagnostic so we always have fresh state for the title
        result = alert_decision(
            OxygenationInput(
                po2=po2, pco2=pco2, pH=ph,
                temperature_c=temperature_c, sensor_po2=sensor_po2,
                hemoglobin_g_dl=hemoglobin,
                venous_sat_percent=venous_sat,
            ),
            yellow_threshold=yellow_threshold,
            orange_threshold=orange_threshold,
            red_threshold=red_threshold,
        )

        # Derive venous PO2 from venous saturation (Hill equation, n=2.7)
        p50_std = effective_p50(pH=ph, pco2=pco2, temp_c=temperature_c)
        svo2 = max(0.01, min(0.99, venous_sat / 100.0))
        P_v_target = float(p50_std * (svo2 / (1.0 - svo2)) ** (1.0 / 2.7))

        numeric_settings = self._get_numeric_settings_inputs()
        if numeric_settings is None:
            return

        self._append(self.t("diag_krogh_computing"))
        self._set_progress_running(True)
        threading.Thread(
            target=self._compute_krogh_reconstruction,
            kwargs={
                "po2": po2, "pco2": pco2, "ph": ph,
                "temperature_c": temperature_c, "sensor_po2": sensor_po2,
                "venous_sat": venous_sat, "P_v_target": P_v_target,
                "venous_weight": venous_weight,
                "diag_result": result,
                "numeric_settings": numeric_settings,
            },
            daemon=True,
        ).start()

    def _compute_krogh_reconstruction(self, po2, pco2, ph, temperature_c, sensor_po2,
                                       venous_sat, P_v_target, venous_weight, diag_result, numeric_settings):
        try:
            with temporary_numeric_settings(numeric_settings):
                fit = self._fit_joint_krogh_parameters(
                    P_inlet=po2,
                    sensor_target=sensor_po2,
                    P_v_target=P_v_target,
                    pH=ph,
                    pCO2=pco2,
                    temp_c=temperature_c,
                    include_axial=True,
                    venous_weight=venous_weight,
                )

            P_half_fit = float(fit["P_half_fit"])
            P_v_sim = float(fit["P_v_sim"])
            p50_eff = float(fit["p50_eff"])
            perfusion_factor = float(fit["perfusion_factor"])
            P_c_axial = fit["P_c_axial"]
            tissue_po2 = np.maximum(fit["tissue_po2"], 0.0)
            sensor_sim = float(fit["sensor_sim"])

            P_avg = np.average(tissue_po2, axis=1, weights=radial_weights)

            x_sym = np.linspace(-R_tis, R_tis, 2 * NR - 1)
            X_sym, Z_sym = np.meshgrid(x_sym, z_eval, indexing="xy")
            Z_rel = Z_sym / L_cap
            R_abs = np.abs(x_sym)
            PO2_sym = np.zeros((NZ, len(x_sym)), dtype=float)
            for i, Pc in enumerate(P_c_axial):
                PO2_sym[i, :] = np.interp(R_abs, r_vec, tissue_po2[i, :], left=Pc, right=tissue_po2[i, -1])
            PO2_sym = np.where(R_abs[None, :] < R_cap, P_c_axial[:, None], PO2_sym)
            PO2_sym = np.maximum(PO2_sym, 0.0)

            plot_data = {
                "po2": po2, "pco2": pco2, "ph": ph, "temperature_c": temperature_c,
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
            }

            fit_info = self.t(
                "diag_krogh_fit_info",
                P_half=P_half_fit,
                sensor_target=sensor_po2,
                sensor_sim=sensor_sim,
            )
            self._append_async(fit_info)
            self._append_async(
                f"[Krogh] perfusion={perfusion_factor:.2f}x | "
                f"P_venous target={P_v_target:.1f} mmHg (simulated={P_v_sim:.1f} mmHg)"
            )
            if float(venous_weight) < 0.3:
                self._append_async(
                    "[Krogh] Optional venous saturation was treated as a weak hint; reconstruction prioritized arterial PO2 and tissue sensor data."
                )
            if bool(fit["fit_warning"]):
                self._append_async(
                    "[Krogh] Note: these diagnostic inputs are only partly representable by a single Krogh cylinder; the plot shows the best joint compromise."
                )
            self._append_async(self.t("diag_krogh_ready"))
            self._call_on_ui_thread(self._show_krogh_reconstruction_plot, plot_data)

        except Exception as exc:
            self._append_async(self.t("diag_krogh_error", error=exc))
        finally:
            self._call_on_ui_thread(self._set_progress_running, False)

    def _show_krogh_reconstruction_plot(self, plot_data):
        dr = diag_result = plot_data["diag_result"]

        # Map 5-state -> normalized alert colour for annotation bar
        _state_colours = {
            "normoxia": "#2ca02c",
            "mild_hypoxia": "#bcbd22",
            "compensated_hypoxia": "#ff7f0e",
            "severe_hypoxia": "#d62728",
            "profound_hypoxia": "#9467bd",
        }
        _alert_colours = {
            "ok": "#2ca02c",
            "alert_yellow": "#e5c011",
            "alert_orange": "#e57c11",
            "critical_alert": "#d62728",
        }
        state = dr["predicted_state"]
        state_label = self._format_oxygenation_state_label(state)
        alert = dr["alert_level"]
        state_colour = _state_colours.get(state, "#333333")
        alert_colour = _alert_colours.get(alert, "#333333")

        po2_mid = float(min(max(plot_data.get("sensor_target", 20.0), 12.0), plot_data["po2_max_plot"] - 1.0))
        if plot_data["po2_min_plot"] < po2_mid < plot_data["po2_max_plot"]:
            po2_norm = TwoSlopeNorm(
                vmin=plot_data["po2_min_plot"],
                vcenter=po2_mid,
                vmax=plot_data["po2_max_plot"],
            )
        else:
            po2_norm = PowerNorm(
                gamma=0.30,
                vmin=plot_data["po2_min_plot"],
                vmax=plot_data["po2_max_plot"],
            )

        fig = plt.figure(figsize=(11, 7.5))

        # Left: 3D surface
        ax3d = fig.add_subplot(121, projection="3d")
        ax3d.plot_surface(
            plot_data["X_sym"] * 1e4,
            plot_data["Z_rel"],
            plot_data["PO2_sym"],
            cmap="coolwarm",
            edgecolor="none",
            alpha=0.95,
            norm=po2_norm,
        )
        contour_levels = np.arange(10, plot_data["po2_max_plot"], 10)
        ax3d.contour(
            plot_data["X_sym"] * 1e4, plot_data["Z_rel"], plot_data["PO2_sym"],
            levels=contour_levels, colors="k", linewidths=0.4,
        )
        ax3d.contourf(
            plot_data["X_sym"] * 1e4, plot_data["Z_rel"], plot_data["PO2_sym"],
            zdir="z", offset=plot_data["po2_min_plot"],
            levels=np.linspace(plot_data["po2_min_plot"], plot_data["po2_max_plot"], 26),
            cmap="coolwarm", alpha=0.50, norm=po2_norm,
        )

        # Capillary walls
        R_cap_um = R_cap * 1e4
        z_rel_vec = z_eval / L_cap
        n = len(z_rel_vec)
        for sign in (+1, -1):
            x_curt = np.full((2, n), sign * R_cap_um)
            y_curt = np.vstack([z_rel_vec, z_rel_vec])
            z_curt = np.vstack([
                np.full(n, plot_data["po2_min_plot"]),
                plot_data["P_c_axial"],
            ])
            top_norm = po2_norm(np.clip(plot_data["P_c_axial"], plot_data["po2_min_plot"], plot_data["po2_max_plot"]))
            facecolors_curt = np.stack([
                plt.cm.coolwarm(po2_norm(np.full(n, plot_data["po2_min_plot"]))),
                plt.cm.coolwarm(top_norm),
            ], axis=0)
            ax3d.plot_surface(x_curt, y_curt, z_curt, facecolors=facecolors_curt, alpha=0.9, shade=False)
            ax3d.plot([sign * R_cap_um] * n, z_rel_vec, plot_data["P_c_axial"], "k-", lw=1.5)

        ax3d.plot(
            np.zeros_like(z_rel_vec), z_rel_vec, plot_data["P_avg"],
            "r-", lw=3.0, label=self.t("legend_sensor_avg"), zorder=15,
        )
        ax3d.set_xlabel(self.t("xlabel_radial_position"))
        ax3d.set_ylabel(self.t("ylabel_relative_length"))
        ax3d.set_zlabel(self.t("zlabel_po2"))
        ax3d.set_xlim(-R_tis * 1e4, R_tis * 1e4)
        ax3d.set_ylim(1.0, 0.0)
        ax3d.set_zlim(plot_data["po2_min_plot"], plot_data["po2_max_plot"])
        ax3d.set_xticks(np.arange(-100, 101, 50))
        ax3d.set_yticks(np.linspace(0, 1, 6))
        ax3d.view_init(elev=24, azim=-57)
        ax3d.xaxis.pane.set_alpha(0.18)
        ax3d.yaxis.pane.set_alpha(0.10)
        ax3d.zaxis.pane.set_alpha(0.0)
        ax3d.grid(False)
        ax3d.legend(loc="upper left", fontsize=8)

        title_3d = self.t(
            "title_3d_diagnostic",
            state=state_label,
            alert=alert.replace("_", " "),
            risk=float(dr.get("risk_score", 0.0)),
            conf=float(dr.get("confidence", 0.0)),
            P_inlet=plot_data["po2"],
            pH=plot_data["ph"],
            pCO2=plot_data["pco2"],
            temp_c=plot_data["temperature_c"],
            P_half=plot_data["P_half_fit"],
            p_venous=plot_data["p_venous"],
            p_tis_min=plot_data["p_tis_min"],
            sensor_avg=plot_data["sensor_avg"],
        )
        ax3d.set_title(title_3d, fontsize=7.5, pad=6)

        # Right: probability bar chart
        ax_bar = fig.add_subplot(122)
        states_ordered = [
            "normoxia", "mild_hypoxia", "compensated_hypoxia",
            "severe_hypoxia", "profound_hypoxia",
        ]
        probs = [float(dr.get(f"p_{s}", 0.0)) for s in states_ordered]
        labels = [self._format_oxygenation_state_label(s).replace(" tissue ", "\n").replace(" ", "\n", 1) for s in states_ordered]
        bar_colours = [_state_colours.get(s, "#888") for s in states_ordered]
        bars = ax_bar.barh(labels, probs, color=bar_colours, edgecolor="k", linewidth=0.6)

        # Highlight the predicted state with a bold border
        pred_idx = states_ordered.index(state) if state in states_ordered else -1
        if pred_idx >= 0:
            bars[pred_idx].set_linewidth(2.5)
            bars[pred_idx].set_edgecolor("black")

        for bar, p in zip(bars, probs):
            ax_bar.text(
                min(p + 0.015, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", ha="left", fontsize=8,
            )

        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_xlabel("Posterior probability")
        ax_bar.set_title(
            f"Tissue oxygenation state probabilities\n"
            f"Alert: {alert.replace('_', ' ')}  "
            f"Risk score: {float(dr.get('risk_score', 0.0)):.3f}",
            fontsize=9, color=alert_colour, fontweight="bold",
        )

        # Sensor comparison annotation
        ax_bar.axvline(x=float(dr.get("risk_score", 0.0)), color=alert_colour,
                       linestyle="--", linewidth=1.2, label=f"risk_score = {float(dr.get('risk_score',0.0)):.3f}")
        ax_bar.legend(fontsize=7, loc="lower right")

        fig.patch.set_facecolor("#f8f8f8")
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])

        if plot_data.get("fit_warning"):
            fig.text(
                0.5, 0.975,
                "Best joint compromise between tissue sensor PO2 and venous saturation",
                ha="center", va="top", fontsize=8, color=alert_colour,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
            )

        venous_label = "venous hint" if float(plot_data.get("venous_weight", 0.0)) < 0.3 else "venous target"

        # Bottom status bar with alert colour
        fig.text(
            0.5, 0.01,
            f"sensor target: {plot_data['sensor_target']:.1f} mmHg | sensor avg: {plot_data['sensor_avg']:.1f} mmHg | "
            f"{venous_label}: {plot_data['P_v_target']:.1f} mmHg | simulated P_venous: {plot_data['P_v_sim']:.1f} mmHg | "
            f"perfusion={plot_data['perfusion_factor']:.2f}x | fitted mitoP50 (P\u00bd): {plot_data['P_half_fit']:.3f} mmHg",
            ha="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=alert_colour, alpha=0.20),
        )

        try:
            fig.canvas.manager.set_window_title(
                f"Krogh Reconstruction — {state_label} ({alert})"
            )
        except Exception:
            pass

        plt.show(block=False)

    def _clear(self):
        self.output.delete("1.0", "end")
        self._set_status(self.t("status_ready"))

    def _quit_application(self):
        if self._is_closing:
            return
        self._is_closing = True
        for tooltip in getattr(self, "numeric_tooltips", []):
            tooltip.destroy()
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            self.quit()
        except (RuntimeError, tk.TclError):
            pass
        try:
            self.destroy()
        except (RuntimeError, tk.TclError):
            pass

    def _run(self):
        self._set_progress_running(True)
        if self.mode_var.get() == "default":
            self._set_status(self.t("status_running_default"))
            threading.Thread(target=self._run_default_script, daemon=True).start()
            return

        if self.mode_var.get() == "series":
            self._set_status(self.t("status_running_series"))
            series_params = self._get_series_inputs()
            if series_params is None:
                self._set_progress_running(False)
                self._set_status(self.t("status_ready"))
                return

            threading.Thread(
                target=self._run_series_worker,
                kwargs=series_params,
                daemon=True,
            ).start()
            return

        params = self._get_single_case_inputs()
        if params is None:
            self._set_progress_running(False)
            self._set_status(self.t("status_ready"))
            return

        numeric_settings = self._get_numeric_settings_inputs()
        if numeric_settings is None:
            self._set_progress_running(False)
            self._set_status(self.t("status_ready"))
            return

        self._set_status(self.t("status_running_single"))

        threading.Thread(
            target=self._run_single_case_worker,
            kwargs={
                **params,
                "numeric_settings": numeric_settings,
                "result_label_context": self._build_result_label_context(params),
            },
            daemon=True,
        ).start()

    def _run_default_script(self):
        script_path = os.path.join(os.path.dirname(__file__), "krogh_basis.py")
        if not os.path.exists(script_path):
            self._append_async(self.t("default_not_found"))
            self._set_status_async(self.t("status_error"))
            self._call_on_ui_thread(self._set_progress_running, False)
            return

        self._append_async(self.t("running_default"))
        success = False
        try:
            proc = subprocess.run(
                [sys.executable, script_path, "--no-show"],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True,
                check=False,
            )
            self._append_async(self.t("return_code", code=proc.returncode))
            if proc.stdout.strip():
                self._append_async(self.t("stdout_last"))
                lines = proc.stdout.splitlines()
                self._append_async("\n".join(lines[-40:]))
            if proc.stderr.strip():
                self._append_async(self.t("stderr"))
                self._append_async(proc.stderr)
            success = True
        except Exception as exc:
            self._append_async(self.t("default_run_error", error=exc))
            self._set_status_async(self.t("status_error"))
        finally:
            if success:
                self._set_status_async(self.t("status_finished"))
            self._call_on_ui_thread(self._set_progress_running, False)
            self._call_on_ui_thread(self._offer_figure_display)

    def _run_single_case_worker(
        self,
        P_inlet,
        P_half,
        pH,
        pCO2,
        temp_c,
        perf,
        high_po2_threshold_primary,
        high_po2_threshold_secondary,
        additional_high_po2_thresholds,
        relative_high_po2_thresholds_percent,
        relative_high_po2_reference,
        include_axial,
        numeric_settings,
        result_label_context,
    ):
        try:
            self._append_async(self.t("running_single"))
            self._set_status_async(self.t("status_running_single"))
            with temporary_numeric_settings(numeric_settings):
                res = run_single_case(
                    P_inlet=P_inlet,
                    P_half=P_half,
                    pH=pH,
                    pCO2=pCO2,
                    temp_c=temp_c,
                    perfusion_factor=perf,
                    include_axial_diffusion=include_axial,
                    high_po2_threshold_primary=high_po2_threshold_primary,
                    high_po2_threshold_secondary=high_po2_threshold_secondary,
                    additional_high_po2_thresholds=additional_high_po2_thresholds,
                    relative_high_po2_thresholds_percent=relative_high_po2_thresholds_percent,
                    relative_high_po2_reference=relative_high_po2_reference,
                )

            self._append_async(self.t("single_result"))
            self._append_async(self.t("inputs_header"))
            self._append_async(
                self.t(
                    "single_inputs_line",
                    P_inlet=P_inlet,
                    P_half=P_half,
                    pH=pH,
                    pCO2=pCO2,
                    temp_c=temp_c,
                    perf=perf,
                    high_po2_threshold_primary=high_po2_threshold_primary,
                    high_po2_threshold_secondary=high_po2_threshold_secondary,
                    relative_po2_reference=self.t(f"reference_{relative_high_po2_reference}"),
                )
            )
            self._append_async(self.t("outputs_header"))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("P50_eff", result_label_context), res["P50_eff"]))
            self._append_async("    {}={:.3f}".format(self._result_label_for_context("Q_flow_nL_s", result_label_context), res["Q_flow_nL_s"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("P_venous", result_label_context), res["P_venous"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("P_tissue_min", result_label_context), res["P_tissue_min"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("P_tissue_p05", result_label_context), res["P_tissue_p05"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("Hypoxic_fraction_lt1", result_label_context), res["Hypoxic_fraction_lt1"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("Hypoxic_fraction_lt5", result_label_context), res["Hypoxic_fraction_lt5"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("Hypoxic_fraction_lt10", result_label_context), res["Hypoxic_fraction_lt10"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_fraction_gt100", result_label_context), res["PO2_fraction_gt100"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_fraction_gt200", result_label_context), res["PO2_fraction_gt200"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_fraction_gt_rel1", result_label_context), res["PO2_fraction_gt_rel1"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_fraction_gt_rel2", result_label_context), res["PO2_fraction_gt_rel2"]))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_fraction_gt_rel3", result_label_context), res["PO2_fraction_gt_rel3"]))
            abs_thresholds = res.get("PO2_absolute_thresholds_mmHg", [])
            abs_values = res.get("PO2_fraction_gt_absolute_all", [])
            for threshold, value in zip(abs_thresholds[2:], abs_values[2:]):
                self._append_async("    {}={:.2f}".format(self.t("result_po2_fraction_gt_primary", threshold=f"{threshold:.6g}"), value))
            rel_thresholds = res.get("PO2_relative_thresholds_percent_of_inlet", [])
            rel_values = res.get("PO2_fraction_gt_relative_all", [])
            for threshold, value in zip(rel_thresholds[3:], rel_values[3:]):
                self._append_async("    {}={:.2f}".format(self.t("result_po2_fraction_gt_rel_primary", threshold=f"{threshold:.6g}"), value))
            self._append_async("    {}={:.2f}".format(self._result_label_for_context("PO2_sensor_avg", result_label_context), res["PO2_sensor_avg"]))
            self._append_async(
                "    {}={:.2f}, {}={:.2f}".format(
                    self._result_label_for_context("S_a_percent", result_label_context),
                    res["S_a_percent"],
                    self._result_label_for_context("S_v_percent", result_label_context),
                    res["S_v_percent"],
                )
            )
            self._append_async("")
            self._set_status_async(self.t("status_finished"))
        except Exception as exc:
            self._append_async(self.t("single_run_error", error=exc))
            self._set_status_async(self.t("status_error"))
        finally:
            self._call_on_ui_thread(self._set_progress_running, False)

    def _run_series_worker(
        self,
        base_params,
        numeric_settings,
        save_bundle_after_display,
        lock_hypoxic_fraction_scale,
        publication_mode,
        publication_layout,
        series_dimension,
        series_plot_mode,
        sweep_field_label,
        start_value,
        end_value,
        step_size,
        secondary_field_label,
        secondary_start_value,
        secondary_end_value,
        secondary_step_size,
        selected_plot_fields,
        result_label_context,
    ):
        try:
            sweep_values = build_series_values(start_value, end_value, step_size)
            secondary_values = None
            if series_dimension == "2d":
                secondary_values = build_series_values(secondary_start_value, secondary_end_value, secondary_step_size)
                self._append_async(
                    self.t(
                        "running_series_2d",
                        parameter1=self._field_label(sweep_field_label),
                        start1=start_value,
                        end1=end_value,
                        parameter2=self._field_label(secondary_field_label),
                        start2=secondary_start_value,
                        end2=secondary_end_value,
                        count1=len(sweep_values),
                        count2=len(secondary_values),
                        count=len(sweep_values) * len(secondary_values),
                    )
                )
            else:
                self._append_async(
                    self.t(
                        "running_series",
                        parameter=self._field_label(sweep_field_label),
                        start=start_value,
                        end=end_value,
                        step=step_size,
                        count=len(sweep_values),
                    )
                )

            case_definitions = build_series_case_definitions(
                base_params,
                sweep_field_label,
                sweep_values,
                secondary_field_label=secondary_field_label if series_dimension == "2d" else None,
                secondary_values=secondary_values,
            )

            with temporary_numeric_settings(numeric_settings):
                results = []
                for case_definition in case_definitions:
                    self._set_status_async(
                        self.t(
                            "status_series_case",
                            index=case_definition["case_index"],
                            count=len(case_definitions),
                        )
                    )
                    if series_dimension == "2d":
                        self._append_async(
                            self.t(
                                "series_case_progress_2d",
                                index=case_definition["case_index"],
                                count=len(case_definitions),
                                parameter1=self._field_label(sweep_field_label),
                                value1=case_definition["sweep_value"],
                                parameter2=self._field_label(secondary_field_label),
                                value2=case_definition["secondary_sweep_value"],
                            )
                        )
                    else:
                        self._append_async(
                            self.t(
                                "series_case_progress",
                                index=case_definition["case_index"],
                                count=len(case_definitions),
                                parameter=self._field_label(sweep_field_label),
                                value=case_definition["sweep_value"],
                            )
                        )
                    results.append(build_series_result_row_from_definition(case_definition))
                    if series_dimension == "2d" and case_definition["case_index"] % len(sweep_values) == 0:
                        outer_index = case_definition["case_index"] // len(sweep_values)
                        outer_value = case_definition["secondary_sweep_value"]
                        progress_text = self.t(
                            "status_series_outer",
                            index=outer_index,
                            count=len(secondary_values),
                            field=self._field_label(secondary_field_label),
                            value=float(outer_value),
                        )
                        self._append_async("  " + progress_text)
                        self._set_status_async(progress_text)

            results_df = pd.DataFrame(results)
            numerics_report = analyze_series_numerics(
                case_definitions,
                results_df,
                selected_plot_fields,
                numeric_settings,
            )
            results_export_df = results_df.copy()
            results_export_df["Sweep_parameter"] = results_export_df["Sweep_parameter"].map(self._field_label)
            results_export_df["Sweep_parameter_2"] = results_export_df["Sweep_parameter_2"].replace("", np.nan)
            results_export_df["Sweep_parameter_2"] = results_export_df["Sweep_parameter_2"].map(
                lambda value: self._field_label(value) if isinstance(value, str) else value
            )
            results_export_df["Include_axial_diffusion"] = results_export_df["Include_axial_diffusion"].map(self._bool_label)
            results_export_df = results_export_df.rename(
                columns={
                    "Case": self.t("result_case"),
                    "Sweep_parameter": self.t("result_sweep_parameter"),
                    "Sweep_value": self.t("result_sweep_value"),
                    "Sweep_parameter_2": self.t("result_sweep_parameter_2"),
                    "Sweep_value_2": self.t("result_sweep_value_2"),
                    "PO2_inlet_mmHg": self._field_label("PO2_inlet_mmHg"),
                    "mitoP50_mmHg": self._field_label("mitoP50_mmHg"),
                    "pH": self._field_label("pH"),
                    "pCO2_mmHg": self._field_label("pCO2_mmHg"),
                    "Temp_C": self._field_label("Temp_C"),
                    "Perfusion_factor": self._field_label("Perfusion_factor"),
                    "High_PO2_threshold_1_mmHg": self._field_label("High_PO2_threshold_1_mmHg"),
                    "High_PO2_threshold_2_mmHg": self._field_label("High_PO2_threshold_2_mmHg"),
                    "High_PO2_additional_thresholds_mmHg": self._field_label("High_PO2_additional_thresholds_mmHg"),
                    "High_PO2_relative_thresholds_percent": self._field_label("High_PO2_relative_thresholds_percent"),
                    "Relative_PO2_reference": self._field_label("Relative_PO2_reference"),
                    "Include_axial_diffusion": self.t("result_include_axial"),
                    "P50_eff": self._result_label_for_context("P50_eff", result_label_context),
                    "P_venous": self._result_label_for_context("P_venous", result_label_context),
                    "P_tissue_min": self._result_label_for_context("P_tissue_min", result_label_context),
                    "P_tissue_p05": self._result_label_for_context("P_tissue_p05", result_label_context),
                    "Hypoxic_fraction_lt1": self._result_label_for_context("Hypoxic_fraction_lt1", result_label_context),
                    "Hypoxic_fraction_lt5": self._result_label_for_context("Hypoxic_fraction_lt5", result_label_context),
                    "Hypoxic_fraction_lt10": self._result_label_for_context("Hypoxic_fraction_lt10", result_label_context),
                    "PO2_fraction_gt100": self._result_label_for_context("PO2_fraction_gt100", result_label_context),
                    "PO2_fraction_gt200": self._result_label_for_context("PO2_fraction_gt200", result_label_context),
                    "PO2_fraction_gt_rel1": self._result_label_for_context("PO2_fraction_gt_rel1", result_label_context),
                    "PO2_fraction_gt_rel2": self._result_label_for_context("PO2_fraction_gt_rel2", result_label_context),
                    "PO2_fraction_gt_rel3": self._result_label_for_context("PO2_fraction_gt_rel3", result_label_context),
                    "PO2_sensor_avg": self._result_label_for_context("PO2_sensor_avg", result_label_context),
                    "S_a_percent": self._result_label_for_context("S_a_percent", result_label_context),
                    "S_v_percent": self._result_label_for_context("S_v_percent", result_label_context),
                    "Q_flow_nL_s": self._result_label_for_context("Q_flow_nL_s", result_label_context),
                }
            )
            setup_df = pd.DataFrame(
                [
                    {self.t("column_setting"): self.t("setting_series_dimension"), self.t("column_value"): self.t(f"series_dimension_{series_dimension}")},
                    {self.t("column_setting"): self.t("setting_plot_mode"), self.t("column_value"): self.t(f"series_plot_mode_{series_plot_mode}")},
                    {self.t("column_setting"): self.t("setting_sweep_parameter"), self.t("column_value"): self._field_label(sweep_field_label)},
                    {self.t("column_setting"): self.t("setting_start_value"), self.t("column_value"): float(start_value)},
                    {self.t("column_setting"): self.t("setting_end_value"), self.t("column_value"): float(end_value)},
                    {self.t("column_setting"): self.t("setting_step_size"), self.t("column_value"): float(step_size)},
                    {self.t("column_setting"): self.t("setting_secondary_sweep_parameter"), self.t("column_value"): "" if not secondary_field_label else self._field_label(secondary_field_label)},
                    {self.t("column_setting"): self.t("setting_secondary_start_value"), self.t("column_value"): "" if secondary_start_value is None else float(secondary_start_value)},
                    {self.t("column_setting"): self.t("setting_secondary_end_value"), self.t("column_value"): "" if secondary_end_value is None else float(secondary_end_value)},
                    {self.t("column_setting"): self.t("setting_secondary_step_size"), self.t("column_value"): "" if secondary_step_size is None else float(secondary_step_size)},
                    {self.t("column_setting"): self.t("setting_case_count"), self.t("column_value"): int(len(case_definitions))},
                    {self.t("column_setting"): self.t("setting_include_axial"), self.t("column_value"): self._bool_label(base_params["include_axial"])},
                    {self.t("column_setting"): self.t("setting_numerics_header"), self.t("column_value"): ""},
                ]
                + [
                    {self.t("column_setting"): self._numeric_label(label_key), self.t("column_value"): numeric_settings[key]}
                    for key, label_key, _ in NUMERIC_SETTINGS_FIELDS
                ]
                + [
                    {self.t("column_setting"): self._field_label(key), self.t("column_value"): value}
                    for key, value in {
                        "PO2_inlet_mmHg": base_params["P_inlet"],
                        "mitoP50_mmHg": base_params["P_half"],
                        "pH": base_params["pH"],
                        "pCO2_mmHg": base_params["pCO2"],
                        "Temp_C": base_params["temp_c"],
                        "Perfusion_factor": base_params["perf"],
                        "High_PO2_threshold_1_mmHg": base_params["high_po2_threshold_primary"],
                        "High_PO2_threshold_2_mmHg": base_params["high_po2_threshold_secondary"],
                        "High_PO2_additional_thresholds_mmHg": ", ".join(f"{value:.6g}" for value in base_params["additional_high_po2_thresholds"]),
                        "High_PO2_relative_thresholds_percent": ", ".join(f"{value:.6g}" for value in base_params["relative_high_po2_thresholds_percent"]),
                        "Relative_PO2_reference": self.t(f"reference_{base_params['relative_high_po2_reference']}") ,
                    }.items()
                ]
            )

            self._append_async(self.t("series_finished"))
            # Re-evaluate a small sample with stricter numerics so the user can see
            # whether the selected sweep is materially sensitive to solver settings.
            self._append_async(self.t("series_check_header", count=numerics_report["sample_count"]))
            for field_name in selected_plot_fields:
                metric = numerics_report["field_metrics"][field_name]
                self._append_async(
                    self.t(
                        "series_check_field",
                        field=self._result_label_for_context(field_name, result_label_context),
                        abs_diff=metric["max_abs_diff"],
                        rel_diff=metric["max_rel_diff"],
                        case=metric["worst_case"],
                    )
                )
            self._append_async(
                self.t("series_check_warning") if numerics_report["has_warning"] else self.t("series_check_ok")
            )
            self._append_async("")
            self._set_status_async(self.t("status_plotting"))
            self._call_on_ui_thread(
                self._show_series_plot,
                results_df,
                sweep_field_label,
                secondary_field_label if series_dimension == "2d" else None,
                selected_plot_fields,
                lock_hypoxic_fraction_scale,
                series_plot_mode,
                results_export_df,
                setup_df,
                save_bundle_after_display,
                publication_mode,
                publication_layout,
                {
                    "series_dimension": series_dimension,
                    "series_plot_mode": series_plot_mode,
                    "publication_mode": publication_mode,
                    "publication_layout": publication_layout,
                    "sweep_field_label": sweep_field_label,
                    "start_value": start_value,
                    "end_value": end_value,
                    "step_size": step_size,
                    "secondary_field_label": secondary_field_label,
                    "secondary_start_value": secondary_start_value,
                    "secondary_end_value": secondary_end_value,
                    "secondary_step_size": secondary_step_size,
                    "selected_plot_fields": selected_plot_fields,
                    "base_params": base_params,
                    "numeric_settings": numeric_settings,
                },
            )
        except Exception as exc:
            self._append_async(self.t("series_run_error", error=exc))
            self._set_status_async(self.t("status_error"))
        finally:
            self._call_on_ui_thread(self._set_progress_running, False)

    def _show_series_plot(
        self,
        results_df,
        sweep_field_label,
        secondary_field_label,
        selected_plot_fields,
        lock_hypoxic_fraction_scale,
        series_plot_mode,
        results_export_df,
        setup_df,
        save_bundle_after_display,
        publication_mode,
        publication_layout,
        bundle_context,
    ):
        x_values = results_df["Sweep_value"].to_numpy()
        x_label = self._field_label(sweep_field_label)
        style = self._get_series_plot_style(publication_mode, publication_layout)
        parameter_text = self._wrap_plot_annotation(
            self._format_series_plot_parameters(results_df, sweep_field_label, secondary_field_label),
            width=style["wrap_width"],
        )
        figures = []
        hypoxic_fields_in_selection = [field for field in selected_plot_fields if field in HYPOXIC_FRACTION_FIELDS]
        shared_hypoxic_ylim = None
        if lock_hypoxic_fraction_scale and hypoxic_fields_in_selection:
            hypoxic_values = np.concatenate(
                [results_df[field].to_numpy(dtype=float) for field in hypoxic_fields_in_selection]
            )
            max_value = float(np.max(hypoxic_values)) if hypoxic_values.size else 0.0
            upper = max(5.0, np.ceil(max_value / 5.0) * 5.0)
            if upper <= 0.0:
                upper = 5.0
            shared_hypoxic_ylim = (0.0, min(100.0, upper))

        if secondary_field_label and series_plot_mode == "3d":
            figures = self._show_series_surface_plots(
                results_df,
                sweep_field_label,
                secondary_field_label,
                selected_plot_fields,
                parameter_text,
                style,
            )
        elif secondary_field_label and series_plot_mode == "heatmap":
            figures = self._show_series_heatmaps(
                results_df,
                sweep_field_label,
                secondary_field_label,
                selected_plot_fields,
                parameter_text,
                style,
            )
        else:
            for field_name in selected_plot_fields:
                fig, ax = plt.subplots(figsize=style["figsize_2d"])
                if secondary_field_label:
                    for secondary_value, subset in results_df.groupby("Sweep_value_2", sort=True):
                        subset = subset.sort_values("Sweep_value")
                        ax.plot(
                            subset["Sweep_value"].to_numpy(dtype=float),
                            subset[field_name].to_numpy(dtype=float),
                            marker="o",
                            linewidth=2.0,
                            label=self.t(
                                "series_curve_legend",
                                field=self._field_label(secondary_field_label),
                                value=float(secondary_value),
                            ),
                        )
                else:
                    y_values = results_df[field_name].to_numpy()
                    ax.plot(x_values, y_values, marker="o", linewidth=2.0)
                ax.set_ylabel(self._result_label(field_name), labelpad=10)
                ax.set_title(self._result_label(field_name), pad=10)
                ax.tick_params(axis="y", pad=6)
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(x_label)
                if shared_hypoxic_ylim is not None and field_name in HYPOXIC_FRACTION_FIELDS:
                    ax.set_ylim(*shared_hypoxic_ylim)
                if secondary_field_label:
                    ax.legend(loc="best", fontsize=style["legend_fontsize"], ncol=2)
                fig.suptitle(self.t("series_plot_title"), y=0.98)
                fig.text(0.5, 0.93, parameter_text, ha="center", va="top", fontsize=style["annotation_fontsize"], linespacing=1.18)
                fig.text(
                    0.5,
                    0.84,
                    self._wrap_plot_annotation(
                        self.t("series_plot_explanation", description=self._result_description(field_name)),
                        width=style["wrap_width"],
                    ),
                    ha="center",
                    va="top",
                    fontsize=style["annotation_fontsize"],
                    linespacing=1.15,
                    wrap=True,
                )
                fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=style["axes_top"])

                try:
                    fig.canvas.manager.set_window_title(
                        self.t("series_plot_window_field", field=self._result_label(field_name))
                    )
                except Exception:
                    pass
                figures.append((field_name, fig))

        plt.show(block=True)

        if save_bundle_after_display:
            selected_dir = filedialog.askdirectory(title=self.t("save_bundle_title"))
            if selected_dir:
                self._save_series_run_bundle(
                    selected_dir,
                    figures,
                    results_export_df,
                    setup_df,
                    bundle_context,
                )
            else:
                self._append(self.t("bundle_save_cancelled"))
        else:
            self._append(self.t("series_results_not_saved"))
            self._append(self.t("series_plots_not_saved"))

        self._set_status(self.t("status_finished"))

    def _show_series_surface_plots(
        self,
        results_df,
        sweep_field_label,
        secondary_field_label,
        selected_plot_fields,
        parameter_text,
        style,
    ):
        figures = []
        for field_name in selected_plot_fields:
            pivot = results_df.pivot(index="Sweep_value_2", columns="Sweep_value", values=field_name)
            x_values = pivot.columns.to_numpy(dtype=float)
            y_values = pivot.index.to_numpy(dtype=float)
            x_grid, y_grid = np.meshgrid(x_values, y_values)
            z_grid = pivot.to_numpy(dtype=float)

            fig = plt.figure(figsize=style["figsize_3d"])
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", edgecolor="none", alpha=0.92)
            ax.set_xlabel(self._field_label(sweep_field_label), labelpad=10)
            ax.set_ylabel(self._field_label(secondary_field_label), labelpad=10)
            ax.set_zlabel(self._result_label(field_name), labelpad=10)
            ax.set_title(self._result_label(field_name), pad=14)
            fig.suptitle(self.t("series_plot_title"), y=0.98)
            fig.text(
                0.5,
                0.93,
                self._wrap_plot_annotation(parameter_text, width=style["wrap_width"]),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.18,
            )
            fig.text(
                0.5,
                0.84,
                self._wrap_plot_annotation(
                    self.t("series_plot_explanation", description=self._result_description(field_name)),
                    width=style["wrap_width"],
                ),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.15,
                wrap=True,
            )
            fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, label=self._result_label(field_name))
            fig.subplots_adjust(left=0.02, right=0.96, bottom=0.08, top=style["axes_top"] + 0.02)

            try:
                fig.canvas.manager.set_window_title(
                    self.t("series_plot_surface_title", field=self._result_label(field_name))
                )
            except Exception:
                pass
            figures.append((field_name, fig))

        return figures

    def _show_series_heatmaps(
        self,
        results_df,
        sweep_field_label,
        secondary_field_label,
        selected_plot_fields,
        parameter_text,
        style,
    ):
        figures = []
        for field_name in selected_plot_fields:
            pivot = results_df.pivot(index="Sweep_value_2", columns="Sweep_value", values=field_name)
            x_values = pivot.columns.to_numpy(dtype=float)
            y_values = pivot.index.to_numpy(dtype=float)
            z_grid = pivot.to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=style["figsize_heatmap"])
            image = ax.imshow(
                z_grid,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values))],
                cmap="viridis",
            )
            ax.set_xlabel(self._field_label(sweep_field_label))
            ax.set_ylabel(self._field_label(secondary_field_label))
            ax.set_title(self._result_label(field_name), pad=10)
            fig.colorbar(image, ax=ax, label=self._result_label(field_name))
            fig.suptitle(self.t("series_plot_title"), y=0.98)
            fig.text(
                0.5,
                0.93,
                self._wrap_plot_annotation(parameter_text, width=style["wrap_width"]),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.18,
            )
            fig.text(
                0.5,
                0.84,
                self._wrap_plot_annotation(
                    self.t("series_plot_explanation", description=self._result_description(field_name)),
                    width=style["wrap_width"],
                ),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.15,
                wrap=True,
            )
            fig.subplots_adjust(left=0.10, right=0.93, bottom=0.12, top=style["axes_top"])

            try:
                fig.canvas.manager.set_window_title(
                    self.t("series_plot_heatmap_title", field=self._result_label(field_name))
                )
            except Exception:
                pass
            figures.append((field_name, fig))

        return figures

    def _save_series_run_bundle(self, parent_dir, figures, results_export_df, setup_df, bundle_context):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(parent_dir, f"krogh_series_run_{timestamp}")
        suffix = 1
        while os.path.exists(run_dir):
            suffix += 1
            run_dir = os.path.join(parent_dir, f"krogh_series_run_{timestamp}_{suffix}")
        os.makedirs(run_dir, exist_ok=False)

        excel_path = os.path.join(run_dir, "series_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            results_export_df.to_excel(writer, sheet_name=self.t("sheet_series_results"), index=False)
            setup_df.to_excel(writer, sheet_name=self.t("sheet_series_setup"), index=False)

        style = self._get_series_plot_style(
            bundle_context.get("publication_mode", False),
            bundle_context.get("publication_layout", "wide"),
        )
        for field_name, fig in figures:
            suffix_mode = ""
            if bundle_context["series_dimension"] == "2d" and bundle_context["series_plot_mode"] == "3d":
                suffix_mode = "_surface"
            elif bundle_context["series_dimension"] == "2d" and bundle_context["series_plot_mode"] == "heatmap":
                suffix_mode = "_heatmap"
            plot_file = os.path.join(run_dir, f"{field_name}{suffix_mode}.png")
            fig.savefig(plot_file, dpi=style["save_dpi"], bbox_inches="tight")
            self._append(self.t("plot_saved", path=plot_file))

        params_path = os.path.join(run_dir, "run_parameters.txt")
        with open(params_path, "w", encoding="utf-8") as handle:
            handle.write(self._format_run_bundle_parameters(bundle_context))

        self._append(self.t("results_saved", path=excel_path))
        self._append(self.t("bundle_file_parameters", path=params_path))
        self._append(self.t("bundle_saved", path=run_dir))

    def _format_run_bundle_parameters(self, bundle_context):
        lines = []
        lines.append("Krogh GUI series run bundle")
        lines.append(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"mode=series")
        lines.append(f"series_dimension={bundle_context['series_dimension']}")
        lines.append(f"series_plot_mode={bundle_context['series_plot_mode']}")
        lines.append(f"publication_mode={bundle_context.get('publication_mode', False)}")
        lines.append(f"publication_layout={bundle_context.get('publication_layout', 'wide')}")
        lines.append(f"sweep_parameter={bundle_context['sweep_field_label']}")
        lines.append(f"start_value={bundle_context['start_value']}")
        lines.append(f"end_value={bundle_context['end_value']}")
        lines.append(f"step_size={bundle_context['step_size']}")
        lines.append(f"secondary_sweep_parameter={bundle_context['secondary_field_label']}")
        lines.append(f"secondary_start_value={bundle_context['secondary_start_value']}")
        lines.append(f"secondary_end_value={bundle_context['secondary_end_value']}")
        lines.append(f"secondary_step_size={bundle_context['secondary_step_size']}")
        lines.append("selected_plot_fields=" + ", ".join(bundle_context["selected_plot_fields"]))
        lines.append("")
        lines.append("base_parameters:")
        for key in sorted(bundle_context["base_params"].keys()):
            lines.append(f"  {key}={bundle_context['base_params'][key]}")
        lines.append("")
        lines.append("numeric_settings:")
        for key in sorted(bundle_context["numeric_settings"].keys()):
            lines.append(f"  {key}={bundle_context['numeric_settings'][key]}")
        lines.append("")
        return "\n".join(lines)

        plt.show(block=False)

    def _offer_figure_display(self):
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krogh_figures")
        if not os.path.isdir(figures_dir):
            return
        png_files = sorted(f for f in os.listdir(figures_dir) if f.endswith("_highres.png"))
        if not png_files:
            return
        fig_names = [f[: -len("_highres.png")] for f in png_files]
        self._show_figure_dialog(fig_names, figures_dir)

    def _show_figure_dialog(self, fig_names, figures_dir):
        dlg = tk.Toplevel(self)
        dlg.title(self.t("show_figures_title"))
        dlg.geometry("520x560")
        dlg.minsize(400, 320)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text=self.t("select_figures"),
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(side="bottom", fill="x", padx=12, pady=8)

        list_frame = ttk.Frame(dlg)
        list_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 4))

        canvas = tk.Canvas(list_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        check_vars = {}
        for name in fig_names:
            var = tk.BooleanVar(value=False)
            check_vars[name] = var
            ttk.Checkbutton(inner, text=name, variable=var).pack(anchor="w", padx=4, pady=2)

        def select_all():
            for value in check_vars.values():
                value.set(True)

        def deselect_all():
            for value in check_vars.values():
                value.set(False)

        def show_selected():
            selected = [name for name, value in check_vars.items() if value.get()]
            dlg.destroy()
            if selected:
                self._display_figures(selected, figures_dir)

        ttk.Button(btn_frame, text=self.t("select_all"), command=select_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=self.t("deselect_all"), command=deselect_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=self.t("show_selected"), command=show_selected).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=self.t("cancel"), command=dlg.destroy).pack(side="right", padx=4)

    def _display_figures(self, fig_names, figures_dir):
        for name in fig_names:
            path = os.path.join(figures_dir, name + "_highres.png")
            if not os.path.exists(path):
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            try:
                fig.canvas.manager.set_window_title(name)
            except Exception:
                pass
            fig.tight_layout(pad=0)
        plt.show(block=False)

    def _run_3d_plot(self):
        if self.mode_var.get() != "single":
            messagebox.showinfo(self.t("info_3d_title"), self.t("info_3d_mode"))
            return

        params = self._get_single_case_inputs()
        if params is None:
            return

        numeric_settings = self._get_numeric_settings_inputs()
        if numeric_settings is None:
            return

        self._append(self.t("plot3d_computing"))
        self._set_progress_running(True)
        threading.Thread(
            target=self._compute_3d_plot_data,
            kwargs={**params, "numeric_settings": numeric_settings},
            daemon=True,
        ).start()

    def _compute_3d_plot_data(
        self,
        P_inlet,
        P_half,
        pH,
        pCO2,
        temp_c,
        perf,
        include_axial,
        numeric_settings,
        **_ignored_kwargs,
    ):
        try:
            with temporary_numeric_settings(numeric_settings):
                p50_eff = effective_p50(pH=pH, pco2=pCO2, temp_c=temp_c)
                P_c_axial, tissue_po2, _ = solve_axial_capillary_po2(
                    P_inlet=P_inlet,
                    P_half=P_half,
                    p50_eff=p50_eff,
                    include_axial_diffusion=include_axial,
                    perfusion_factor=perf,
                )
            P_avg = np.average(tissue_po2, axis=1, weights=radial_weights)

            x_sym = np.linspace(-R_tis, R_tis, 2 * NR - 1)
            X_sym, Z_sym = np.meshgrid(x_sym, z_eval, indexing="xy")
            Z_rel = Z_sym / L_cap
            R_abs = np.abs(x_sym)
            PO2_sym = np.zeros((NZ, len(x_sym)), dtype=float)
            for i, Pc in enumerate(P_c_axial):
                PO2_sym[i, :] = np.interp(R_abs, r_vec, tissue_po2[i, :], left=Pc, right=tissue_po2[i, -1])
            PO2_sym = np.where(R_abs[None, :] < R_cap, P_c_axial[:, None], PO2_sym)
            PO2_sym = np.maximum(PO2_sym, 0.0)

            plot_data = {
                "P_inlet": P_inlet,
                "P_half": P_half,
                "pH": pH,
                "pCO2": pCO2,
                "temp_c": temp_c,
                "perf": perf,
                "p50_eff": float(p50_eff),
                "P_c_axial": P_c_axial,
                "P_avg": P_avg,
                "X_sym": X_sym,
                "Z_rel": Z_rel,
                "PO2_sym": PO2_sym,
                "po2_min_plot": 0.0,
                "po2_max_plot": max(float(P_inlet) * 1.05, 30.0),
                "p_venous": float(P_c_axial[-1]),
                "p_tis_min": float(np.min(np.maximum(tissue_po2, 0.0))),
                "sensor_avg": float(P_avg.mean()),
            }

            self._append_async(self.t("plot3d_ready"))
            self._call_on_ui_thread(self._show_3d_plot, plot_data)
        except Exception as exc:
            self._append_async(self.t("plot3d_error", error=exc))
        finally:
            self._call_on_ui_thread(self._set_progress_running, False)

    def _show_3d_plot(self, plot_data):
        clinical_center = float(min(max(35.0, plot_data["po2_min_plot"] + 5.0), plot_data["po2_max_plot"] - 1.0))
        if plot_data["po2_min_plot"] < clinical_center < plot_data["po2_max_plot"]:
            po2_norm = TwoSlopeNorm(
                vmin=plot_data["po2_min_plot"],
                vcenter=clinical_center,
                vmax=plot_data["po2_max_plot"],
            )
        else:
            po2_norm = PowerNorm(
                gamma=0.30,
                vmin=plot_data["po2_min_plot"],
                vmax=plot_data["po2_max_plot"],
            )

        fig = plt.figure(figsize=(10, 7))
        ax3d = fig.add_subplot(111, projection="3d")

        surf = ax3d.plot_surface(
            plot_data["X_sym"] * 1e4,
            plot_data["Z_rel"],
            plot_data["PO2_sym"],
            cmap="coolwarm",
            edgecolor="none",
            alpha=0.95,
            norm=po2_norm,
        )

        contour_levels = np.arange(10, plot_data["po2_max_plot"], 10)
        ax3d.contour(
            plot_data["X_sym"] * 1e4,
            plot_data["Z_rel"],
            plot_data["PO2_sym"],
            levels=contour_levels,
            colors="k",
            linewidths=0.5,
        )
        ax3d.contourf(
            plot_data["X_sym"] * 1e4,
            plot_data["Z_rel"],
            plot_data["PO2_sym"],
            zdir="z",
            offset=plot_data["po2_min_plot"],
            levels=np.linspace(plot_data["po2_min_plot"], plot_data["po2_max_plot"], 26),
            cmap="coolwarm",
            alpha=0.55,
            norm=po2_norm,
        )

        R_cap_um = R_cap * 1e4
        z_rel_vec = z_eval / L_cap
        n = len(z_rel_vec)
        for sign in (+1, -1):
            x_curt = np.full((2, n), sign * R_cap_um)
            y_curt = np.vstack([z_rel_vec, z_rel_vec])
            z_curt = np.vstack([
                np.full(n, plot_data["po2_min_plot"]),
                plot_data["P_c_axial"],
            ])
            top_norm = po2_norm(np.clip(plot_data["P_c_axial"], plot_data["po2_min_plot"], plot_data["po2_max_plot"]))
            facecolors_curt = np.stack(
                [
                    plt.cm.coolwarm(po2_norm(np.full(n, plot_data["po2_min_plot"]))),
                    plt.cm.coolwarm(top_norm),
                ],
                axis=0,
            )
            ax3d.plot_surface(x_curt, y_curt, z_curt, facecolors=facecolors_curt, alpha=0.9, shade=False)
            ax3d.plot([sign * R_cap_um] * n, z_rel_vec, plot_data["P_c_axial"], "k-", lw=1.5)

        ax3d.plot(
            np.zeros_like(z_rel_vec),
            z_rel_vec,
            plot_data["P_avg"],
            "r-",
            lw=3.0,
            label=self.t("legend_sensor_avg"),
            zorder=15,
        )

        ax3d.set_xlabel(self.t("xlabel_radial_position"))
        ax3d.set_ylabel(self.t("ylabel_relative_length"))
        ax3d.set_zlabel(self.t("zlabel_po2"))
        ax3d.set_xlim(-R_tis * 1e4, R_tis * 1e4)
        ax3d.set_ylim(1.0, 0.0)
        ax3d.set_zlim(plot_data["po2_min_plot"], plot_data["po2_max_plot"])
        ax3d.set_xticks(np.arange(-100, 101, 50))
        ax3d.set_yticks(np.linspace(0, 1, 6))
        ax3d.view_init(elev=24, azim=-57)
        ax3d.xaxis.pane.set_alpha(0.18)
        ax3d.yaxis.pane.set_alpha(0.10)
        ax3d.zaxis.pane.set_alpha(0.0)
        ax3d.grid(False)
        ax3d.legend(loc="upper left", fontsize=9)
        ax3d.set_title(
            self.t(
                "title_3d",
                P_inlet=plot_data["P_inlet"],
                P_half=plot_data["P_half"],
                pH=plot_data["pH"],
                pCO2=plot_data["pCO2"],
                temp_c=plot_data["temp_c"],
                perf=plot_data["perf"],
                p50_eff=plot_data["p50_eff"],
                p_venous=plot_data["p_venous"],
                p_tis_min=plot_data["p_tis_min"],
                sensor_avg=plot_data["sensor_avg"],
            ),
            fontsize=9,
        )

        cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.018])
        fig.colorbar(surf, cax=cbar_ax, orientation="horizontal", label=self.t("colorbar_po2"))
        fig.subplots_adjust(bottom=0.10)
        plt.show(block=False)

    def _save_case(self):
        data = {name: entry.get() for name, entry in self.entries.items()}
        data["include_axial_diffusion"] = self.include_axial_var.get()
        data["numeric_settings"] = {
            key: entry.get() for key, entry in self.numeric_entries.items()
        }
        data["diagnostic_settings"] = {
            key: entry.get() for key, entry in self.diagnostic_entries.items()
        }
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[(self.t("json_files"), "*.json"), (self.t("all_files"), "*")],
            title=self.t("save_case_title"),
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._append(self.t("case_saved_to", path=path))
        except Exception as exc:
            messagebox.showerror(self.t("save_error_title"), str(exc))

    def _load_case(self):
        path = filedialog.askopenfilename(
            filetypes=[(self.t("json_files"), "*.json"), (self.t("all_files"), "*")],
            title=self.t("load_case_title"),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror(self.t("load_error_title"), str(exc))
            return

        self.mode_var.set("single")
        self._toggle_inputs()

        for name, entry in self.entries.items():
            if name in data:
                entry.config(state="normal")
                entry.delete(0, "end")
                entry.insert(0, str(data[name]))

        if "include_axial_diffusion" in data:
            self.include_axial_var.set(bool(data["include_axial_diffusion"]))

        if "numeric_settings" in data and isinstance(data["numeric_settings"], dict):
            for key, entry in self.numeric_entries.items():
                if key in data["numeric_settings"]:
                    entry.config(state="normal")
                    entry.delete(0, "end")
                    entry.insert(0, str(data["numeric_settings"][key]))

        if "diagnostic_settings" in data and isinstance(data["diagnostic_settings"], dict):
            for key, entry in self.diagnostic_entries.items():
                if key in data["diagnostic_settings"]:
                    entry.config(state="normal")
                    entry.delete(0, "end")
                    entry.insert(0, str(data["diagnostic_settings"][key]))

        self._append(self.t("case_loaded_from", path=path))


if __name__ == "__main__":
    app = KroghGUI()
    app.mainloop()
