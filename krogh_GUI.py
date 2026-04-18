"""Tkinter GUI for running Krogh cylinder single-case and series analyses.

Recent maintenance notes:
- Numerical solver controls for the capillary ODE and axial coupling are exposed in
    the GUI and are persisted in saved case files.
- Default capillary solver tolerances and maximum step size were tightened to reduce
    non-physical artifacts in sensitive sweeps such as mitoP50 and low-perfusion runs.
- Series runs report a sampled stricter-solver comparison so numerically sensitive
    parameter regions can be spotted without re-running the full sweep manually.
"""

import os
import sys
import threading
import subprocess
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

from krogh_app.constants import (
    AXIAL_COUPLING_MAX_ITER,
    AXIAL_COUPLING_TOL,
    AXIAL_DIFFUSION_MAX_ITER,
    AXIAL_DIFFUSION_RELAX,
    AXIAL_DIFFUSION_TOL,
    BOHR_COEFF,
    CAPILLARY_ODE_ATOL,
    CAPILLARY_ODE_MAX_STEP,
    CAPILLARY_ODE_RTOL,
    CO2_COEFF,
    C_Hb,
    K_diff,
    L_cap,
    MIN_CONSUMPTION_FRACTION,
    M_rate,
    NR,
    NUMERIC_SETTINGS_FIELDS,
    NUMERIC_SETTINGS_TYPES,
    NUMERIC_SETTING_SPECS,
    NZ,
    P50,
    PCO2_REF,
    PH_REF,
    Q_flow,
    R_cap,
    R_tis,
    TEMP_COEFF,
    TEMP_REF,
    alpha,
    dr,
    dz,
    n_hill,
    r_vec,
    radial_weights,
    v_blood,
    z_eval,
)
from krogh_app.diagnostics import DiagnosticEngine
from krogh_app.helptext import HelpTextBuilder
from krogh_app.localization import TranslationManager
from krogh_app.persistence import CaseRepository
from krogh_app.plotting import PlotManager, PlotWorkflowCoordinator
from krogh_app.reconstruction import KroghReconstructor
from krogh_app.series import SeriesRunner
from krogh_app.types import DiagnosticRunInput, NumericSettings
from krogh_app.ui.controls import UIControlCoordinator
from krogh_app.ui.dialogs import show_scrolled_text_dialog
from krogh_app.ui.execution import UIExecutionCoordinator
from krogh_app.ui.figures import UIFigureCoordinator
from krogh_app.ui.layout import UIWindowBuilder
from krogh_app.ui.runtime import UIRuntimeCoordinator
from krogh_app.ui.tooltips import ToolTip

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# -----------------------------
# Shared model constants
# -----------------------------
# Initial defaults now live in krogh_app.constants and are imported above.
# The values remain mutable in this module so the existing GUI workflow and
# numeric-settings context manager keep working unchanged.

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
        "save_diagnostic_report_button": "Save diagnostic report...",
        "save_publication_report_button": "Save English publication report...",
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
        "diag_export_no_result": "Run diagnostic first before exporting a report.",
        "save_diagnostic_report_title": "Save diagnostic report...",
        "save_publication_report_title": "Save English publication report...",
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
        "save_diagnostic_report_button": "Diagnostikbericht speichern...",
        "save_publication_report_button": "Englischen Publikationsbericht speichern...",
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
        "diag_export_no_result": "Bitte zuerst die Diagnostik ausfuehren, bevor ein Bericht exportiert wird.",
        "save_diagnostic_report_title": "Diagnostikbericht speichern...",
        "save_publication_report_title": "Englischen Publikationsbericht speichern...",
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
        "save_diagnostic_report_button": "Enregistrer rapport diagnostique...",
        "save_publication_report_button": "Enregistrer rapport scientifique en anglais...",
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
        "diag_export_no_result": "Executer d'abord le diagnostic avant d'exporter un rapport.",
        "save_diagnostic_report_title": "Enregistrer rapport diagnostique...",
        "save_publication_report_title": "Enregistrer rapport scientifique en anglais...",
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
        "diag_export_no_result": "Eseguire prima la diagnostica prima di esportare un report.",
        "save_diagnostic_report_title": "Salva report diagnostico...",
        "save_publication_report_title": "Salva report scientifico in inglese...",
        "diag_result_line": "Stato={state} | punteggio_rischio={risk_score:.3f} | confidenza={confidence:.3f} | certezza={certainty:.3f} | allarme={alert}",
        "run_diagnostic_button": "Esegui diagnostica",
        "save_diagnostic_template_button": "Salva modello di calibrazione...",
        "save_diagnostic_report_button": "Salva report diagnostico...",
        "save_publication_report_button": "Salva report scientifico in inglese...",
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
        "save_diagnostic_report_button": "Guardar informe diagnostico...",
        "save_publication_report_button": "Guardar informe cientifico en ingles...",
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
        "diag_export_no_result": "Ejecuta primero el diagnostico antes de exportar un informe.",
        "save_diagnostic_report_title": "Guardar informe diagnostico...",
        "save_publication_report_title": "Guardar informe cientifico en ingles...",
        "diag_result_line": "Estado={state} | puntuacion_riesgo={risk_score:.3f} | confianza={confidence:.3f} | certeza={certainty:.3f} | alerta={alert}",
    },
}


def translate(language_code, key, **kwargs):
    language_map = TRANSLATIONS.get(language_code, TRANSLATIONS["en"])
    template = language_map.get(key, TRANSLATIONS["en"].get(key, key))
    return template.format(**kwargs)


def get_numeric_settings():
    return NumericSettings(
        ode_rtol=float(CAPILLARY_ODE_RTOL),
        ode_atol=float(CAPILLARY_ODE_ATOL),
        ode_max_step=float(CAPILLARY_ODE_MAX_STEP),
        axial_diffusion_max_iter=int(AXIAL_DIFFUSION_MAX_ITER),
        axial_diffusion_tol=float(AXIAL_DIFFUSION_TOL),
        axial_coupling_max_iter=int(AXIAL_COUPLING_MAX_ITER),
        axial_coupling_tol=float(AXIAL_COUPLING_TOL),
    ).to_dict()


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


def _get_series_runner():
    return SeriesRunner(
        run_single_case=run_single_case,
        series_sweep_fields=SERIES_SWEEP_FIELDS,
        build_tighter_numeric_settings=build_tighter_numeric_settings,
        build_series_check_indices=build_series_check_indices,
        temporary_numeric_settings=temporary_numeric_settings,
    )


def build_series_values(start_value, end_value, step_size):
    return _get_series_runner().build_values(start_value, end_value, step_size)


def build_series_case_definitions(
    base_params,
    sweep_field_label,
    sweep_values,
    secondary_field_label=None,
    secondary_values=None,
):
    return _get_series_runner().build_case_definitions(
        base_params,
        sweep_field_label,
        sweep_values,
        secondary_field_label=secondary_field_label,
        secondary_values=secondary_values,
    )


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
    return _get_series_runner().build_result_row(
        case_index,
        sweep_field_label,
        sweep_value,
        case_params,
        secondary_field_label=secondary_field_label,
        secondary_sweep_value=secondary_sweep_value,
    )


def build_series_result_row_from_definition(case_definition):
    return _get_series_runner().build_result_row_from_definition(case_definition)


def analyze_series_numerics(case_definitions, results_df, selected_fields, numeric_settings):
    return _get_series_runner().analyze_numerics(case_definitions, results_df, selected_fields, numeric_settings)


class KroghGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self._main_thread_ident = threading.get_ident()
        self.language_code = "en"
        self.translation_manager = TranslationManager(
            translations=TRANSLATIONS,
            input_field_labels=INPUT_FIELD_LABELS,
            result_field_labels=SERIES_RESULT_FIELDS,
        )
        self.language_display_var = tk.StringVar(value=LANGUAGE_NAMES[self.language_code])
        self.title(self.t("app_title"))
        self.geometry("1080x860")
        self.minsize(980, 720)
        self._is_closing = False
        self.protocol("WM_DELETE_WINDOW", self._quit_application)
        self.diagnostic_engine = DiagnosticEngine()
        self.help_text_builder = HelpTextBuilder()
        self.case_repository = CaseRepository()
        self.ui_runtime = UIRuntimeCoordinator()
        self.ui_controls = UIControlCoordinator()
        self.ui_execution = UIExecutionCoordinator(project_dir=CURRENT_DIR)
        self.ui_figures = UIFigureCoordinator(project_dir=CURRENT_DIR)
        self.ui_builder = UIWindowBuilder(
            language_names=LANGUAGE_NAMES,
            series_sweep_fields=SERIES_SWEEP_FIELDS,
            series_result_fields=SERIES_RESULT_FIELDS,
            get_numeric_settings=get_numeric_settings,
        )
        self.plot_manager = PlotManager()
        self.last_diagnostic_result = None
        self.last_krogh_reconstruction = None
        self.plot_workflow = PlotWorkflowCoordinator(
            hypoxic_fields=HYPOXIC_FRACTION_FIELDS,
            r_cap=R_cap,
            r_tis=R_tis,
        )
        self.reconstructor = KroghReconstructor(
            solve_axial_capillary_po2=solve_axial_capillary_po2,
            effective_p50=effective_p50,
            radial_weights=radial_weights,
            r_vec=r_vec,
            z_eval=z_eval,
            R_cap=R_cap,
            R_tis=R_tis,
            L_cap=L_cap,
        )
        self.series_runner = _get_series_runner()

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
        self.diagnostic_radius_mode_var = tk.StringVar(value="all variants")
        self.diagnostic_radius_variant_var = tk.StringVar(value="100 µm")

        self._build_ui()

    def t(self, key, **kwargs):
        return self.translation_manager.translate(key, language_code=self.language_code, **kwargs)

    def _field_label(self, field_name):
        return self.translation_manager.field_label(field_name, language_code=self.language_code)

    def _get_diagnostic_radius_preferences(self):
        mode_raw = str(self.diagnostic_radius_mode_var.get() or "all variants").strip().lower()
        mode = "selected" if "selected" in mode_raw else "all"
        label = str(self.diagnostic_radius_variant_var.get() or "100 µm").strip()
        key_map = {
            "30 µm": "normal_30um",
            "50 µm": "increased_50um",
            "100 µm": "high_100um",
        }
        return mode, key_map.get(label, "high_100um"), label

    def _toggle_diagnostic_radius_variant_controls(self, _event=None):
        if hasattr(self, "diagnostic_radius_variant_combo"):
            mode, _, _ = self._get_diagnostic_radius_preferences()
            self.diagnostic_radius_variant_combo.configure(state="readonly" if mode == "selected" else "disabled")

    def _format_radius_alert_summary(self, plot_data, mode, selected_key):
        scenarios = plot_data.get("radius_scenarios", {}) or {}
        if not scenarios:
            return ""

        ordered_keys = ["normal_30um", "increased_50um", "high_100um"]
        if mode == "selected":
            chosen = scenarios.get(selected_key) or scenarios.get("high_100um") or next(iter(scenarios.values()))
            return (
                f"Radius-conditioned alert under {float(chosen.get('radius_um', 100.0)):.0f} µm assumption: "
                f"{chosen.get('alert_level', 'unknown')} — {chosen.get('interpretation', '')}."
            )

        parts = []
        for key in ordered_keys:
            if key in scenarios:
                scenario = scenarios[key]
                parts.append(f"{float(scenario.get('radius_um', 0.0)):.0f} µm={scenario.get('alert_level', 'unknown')}")
        base_summary = plot_data.get("radius_sensitivity_summary", "")
        return f"Radius-conditioned alerts: {' | '.join(parts)}. {base_summary}".strip()

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
        return self.translation_manager.result_label(
            field_name,
            language_code=self.language_code,
            absolute_thresholds=tuple(float(value) for value in absolute_thresholds[:2]),
            relative_thresholds=tuple(float(value) for value in relative_thresholds[:3]),
            relative_reference=str(relative_reference),
        )

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
        return self.translation_manager.parse_relative_reference_mode(
            raw_value,
            language_code=self.language_code,
        )

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
        return self.translation_manager.result_description(
            field_name,
            language_code=self.language_code,
        )

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
        return self.plot_manager.wrap_annotation(text, width=width)

    def _get_series_plot_style(self, publication_mode=False, publication_layout="wide"):
        return self.plot_manager.get_series_plot_style(
            publication_mode=publication_mode,
            publication_layout=publication_layout,
        )

    def _build_output_parameter_help_text(self):
        return self.help_text_builder.build_output_parameter_help_text(
            language_code=self.language_code,
            t=self.t,
            result_label=self._result_label,
        )

    def _build_numeric_parameter_help_text(self):
        return self.help_text_builder.build_numeric_parameter_help_text(
            language_code=self.language_code,
            t=self.t,
            numeric_settings_fields=NUMERIC_SETTINGS_FIELDS,
            get_numeric_spec=self._get_numeric_spec,
            numeric_label=self._numeric_label,
            format_numeric_value=self._format_numeric_value,
            current_value_getter=lambda key: self.numeric_entries[key].get().strip(),
        )

    def _show_output_parameter_help(self):
        show_scrolled_text_dialog(
            self,
            title=self.t("output_help_title"),
            content=self._build_output_parameter_help_text(),
            button_text=self.t("cancel"),
            geometry="860x620",
        )

    def _show_numeric_parameter_help(self):
        show_scrolled_text_dialog(
            self,
            title=self.t("numeric_help_title"),
            content=self._build_numeric_parameter_help_text(),
            button_text=self.t("cancel"),
            geometry="860x620",
        )

    def _numeric_label(self, field_name):
        return self.translation_manager.numeric_label(field_name, language_code=self.language_code)

    def _bool_label(self, value):
        return self.translation_manager.bool_label(bool(value), language_code=self.language_code)

    def _capture_ui_state(self):
        return self.case_repository.capture_ui_state(self)

    def _restore_ui_state(self, state):
        self.case_repository.restore_ui_state(self, state)

    def _on_language_selected(self, _event=None):
        self.ui_controls.apply_language_selection(self)

    def _set_series_param_display(self, field_key):
        self.ui_controls.set_series_param_display(self, field_key)

    def _set_series_param2_display(self, field_key):
        self.ui_controls.set_series_param2_display(self, field_key)

    def _set_publication_layout_display(self, layout_key):
        self.ui_controls.set_publication_layout_display(self, layout_key)

    def _on_publication_layout_selected(self, _event=None):
        self.ui_controls.on_publication_layout_selected(self)

    def _toggle_series_dimension_inputs(self):
        self.ui_controls.toggle_series_dimension_inputs(self)

    def _build_ui(self):
        self.ui_builder.build(self)

    def _call_on_ui_thread(self, callback, *args, **kwargs):
        self.ui_runtime.call_on_ui_thread(self, callback, *args, **kwargs)

    def _append(self, text):
        self.ui_runtime.append_output(self, text)

    def _append_async(self, text):
        self.ui_runtime.append_output_async(self, text)

    def _set_status(self, text):
        self.ui_runtime.set_status(self, text)

    def _set_status_async(self, text):
        self.ui_runtime.set_status_async(self, text)

    def _set_progress_running(self, running):
        self.ui_runtime.set_progress_running(self, running)

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
        return NumericSettings(**settings).to_dict()

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
        if hasattr(self, "save_diagnostic_report_button"):
            self.save_diagnostic_report_button.config(state=state)
        if hasattr(self, "save_publication_report_button"):
            self.save_publication_report_button.config(state=state)
        if hasattr(self, "reconstruct_krogh_button"):
            self.reconstruct_krogh_button.config(state=state)

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
        language = self.language_display_var.get() if hasattr(self, "language_display_var") else self.language_code
        return self.translation_manager.format_oxygenation_state_label(
            state_name,
            language_code=language,
        )

    def _run_diagnostic_from_inputs(self):
        if not self.diagnostic_engine.is_available:
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

        if not self.diagnostic_engine.validate_thresholds(yellow_threshold, orange_threshold, red_threshold):
            messagebox.showerror(self.t("input_error_title"), "Thresholds must satisfy 0 ≤ yellow ≤ orange ≤ red ≤ 1.0")
            return

        result = self.diagnostic_engine.evaluate(
            DiagnosticRunInput(
                po2=po2,
                pco2=pco2,
                pH=ph,
                temperature_c=temperature_c,
                sensor_po2=sensor_po2,
                hemoglobin_g_dl=hemoglobin,
                venous_sat_percent=venous_sat,
                yellow_threshold=yellow_threshold,
                orange_threshold=orange_threshold,
                red_threshold=red_threshold,
            )
        )
        self.last_diagnostic_result = dict(result)
        self.last_krogh_reconstruction = None

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
        radius_mode, _, radius_label = self._get_diagnostic_radius_preferences()
        if radius_mode == "selected":
            output_lines.append(f"Selected radius-conditioned interpretation: {radius_label}")
        else:
            output_lines.append("Radius-conditioned interpretation can compare all variants: 30 µm, 50 µm, and 100 µm.")
        if result.get("driver_summary"):
            output_lines.append("")
            output_lines.append(result["driver_summary"])
            output_lines.append("Run Krogh reconstruction to estimate hidden tissue burden and compare 30, 50, and 100 µm cylinder assumptions.")
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
                    "sensor_po2": float(self.diagnostic_entries["sensor_po2"].get() or 25.0),
                    "true_state": "",
                }
            ]
        )
        try:
            template.to_csv(path, index=False)
            self._append(self.t("results_saved", path=path))
        except Exception as exc:
            messagebox.showerror(self.t("save_error_title"), str(exc))

    def _save_diagnostic_report(self):
        if not self.last_diagnostic_result:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_export_no_result"))
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                (self.t("all_files"), "*.*"),
            ],
            title=self.t("save_diagnostic_report_title"),
        )
        if not path:
            return

        try:
            report = self.case_repository.build_diagnostic_report(self)
            self.case_repository.save_diagnostic_report(report, path)
            self._append(self.t("results_saved", path=path))
        except Exception as exc:
            messagebox.showerror(self.t("save_error_title"), str(exc))

    def _save_publication_report(self):
        if not self.last_diagnostic_result:
            messagebox.showerror(self.t("input_error_title"), self.t("diag_export_no_result"))
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("LaTeX files", "*.tex"),
                ("Text files", "*.txt"),
                (self.t("all_files"), "*.*"),
            ],
            title=self.t("save_publication_report_title"),
        )
        if not path:
            return

        try:
            if str(path).lower().endswith(".txt"):
                content = self.case_repository.build_publication_report_text(self)
            else:
                content = self.case_repository.build_publication_report_tex(self)
            self.case_repository.save_publication_report(content, path)
            self._append(self.t("results_saved", path=path))
        except Exception as exc:
            messagebox.showerror(self.t("save_error_title"), str(exc))

    # ------------------------------------------------------------------
    # Krogh reconstruction from diagnostic result
    # ------------------------------------------------------------------

    def _fit_p_half_from_venous(self, P_inlet, P_v_target, pH, pCO2, temp_c,
                                 perfusion_factor=1.0, include_axial=True):
        return self.reconstructor.fit_p_half_from_venous(
            P_inlet=P_inlet,
            P_v_target=P_v_target,
            pH=pH,
            pCO2=pCO2,
            temp_c=temp_c,
            perfusion_factor=perfusion_factor,
            include_axial=include_axial,
        )

    def _fit_joint_krogh_parameters(self, P_inlet, sensor_target, P_v_target, pH, pCO2, temp_c,
                                    include_axial=True, venous_weight=0.15):
        return self.reconstructor.fit_joint_parameters(
            P_inlet=P_inlet,
            sensor_target=sensor_target,
            P_v_target=P_v_target,
            pH=pH,
            pCO2=pCO2,
            temp_c=temp_c,
            include_axial=include_axial,
            venous_weight=venous_weight,
        )

    def _run_reconstruct_krogh(self):
        """Triggered by the 'Reconstruct Krogh Cylinder' button."""
        if not self.diagnostic_engine.is_available:
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
        result = self.diagnostic_engine.evaluate(
            DiagnosticRunInput(
                po2=po2,
                pco2=pco2,
                pH=ph,
                temperature_c=temperature_c,
                sensor_po2=sensor_po2,
                hemoglobin_g_dl=hemoglobin,
                venous_sat_percent=venous_sat,
                yellow_threshold=yellow_threshold,
                orange_threshold=orange_threshold,
                red_threshold=red_threshold,
            )
        )

        # Derive venous PO2 from venous saturation through the shared diagnostic service
        P_v_target = self.diagnostic_engine.venous_target_po2(
            pH=ph,
            pco2=pco2,
            temperature_c=temperature_c,
            venous_sat_percent=venous_sat,
        )

        numeric_settings = self._get_numeric_settings_inputs()
        if numeric_settings is None:
            return

        self._append(self.t("diag_krogh_computing"))
        self._set_progress_running(True)
        radius_mode, selected_radius_key, selected_radius_label = self._get_diagnostic_radius_preferences()
        threading.Thread(
            target=self._compute_krogh_reconstruction,
            kwargs={
                "po2": po2, "pco2": pco2, "ph": ph,
                "temperature_c": temperature_c, "sensor_po2": sensor_po2,
                "venous_sat": venous_sat, "P_v_target": P_v_target,
                "venous_weight": venous_weight,
                "diag_result": result,
                "numeric_settings": numeric_settings,
                "radius_mode": radius_mode,
                "selected_radius_key": selected_radius_key,
                "selected_radius_label": selected_radius_label,
            },
            daemon=True,
        ).start()

    def _compute_krogh_reconstruction(self, po2, pco2, ph, temperature_c, sensor_po2,
                                       venous_sat, P_v_target, venous_weight, diag_result, numeric_settings,
                                       radius_mode="all", selected_radius_key="high_100um", selected_radius_label="100 µm"):
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

            plot_data = self.reconstructor.build_plot_data(
                po2=po2,
                pco2=pco2,
                ph=ph,
                temperature_c=temperature_c,
                venous_sat=venous_sat,
                P_v_target=P_v_target,
                venous_weight=venous_weight,
                sensor_po2=sensor_po2,
                diag_result=diag_result,
                fit=fit,
            )

            radius_alert_summary = self._format_radius_alert_summary(plot_data, radius_mode, selected_radius_key)
            if radius_alert_summary:
                plot_data["radius_sensitivity_summary"] = radius_alert_summary

            P_half_fit = float(plot_data["P_half_fit"])
            P_v_sim = float(plot_data["P_v_sim"])
            perfusion_factor = float(plot_data["perfusion_factor"])
            sensor_sim = float(plot_data["sensor_sim"])
            self.last_diagnostic_result = dict(diag_result)
            self.last_krogh_reconstruction = {
                "P_half_fit": P_half_fit,
                "P_v_target": float(P_v_target),
                "P_v_sim": P_v_sim,
                "perfusion_factor": perfusion_factor,
                "sensor_target": float(sensor_po2),
                "sensor_sim": sensor_sim,
                "sensor_error": float(plot_data.get("sensor_error", 0.0)),
                "venous_error": float(plot_data.get("venous_error", 0.0)),
                "fit_warning": bool(plot_data.get("fit_warning", False)),
                "uncertainty": dict(plot_data.get("uncertainty", {})),
                "hypoxic_fraction_map": dict(plot_data.get("hypoxic_fraction_map", {})),
                "hypoxic_burden_summary": str(plot_data.get("hypoxic_burden_summary", "") or ""),
                "radius_scenarios": dict(plot_data.get("radius_scenarios", {})),
                "radius_mode": str(radius_mode),
                "selected_radius_key": str(selected_radius_key),
                "selected_radius_label": str(selected_radius_label),
                "radius_sensitivity_summary": str(plot_data.get("radius_sensitivity_summary", "") or ""),
                "assumption_summary": str(plot_data.get("assumption_summary", "") or ""),
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
            if plot_data.get("assumption_summary"):
                self._append_async(f"[Krogh] assumption summary: {plot_data['assumption_summary']}")
            if plot_data.get("hypoxic_burden_summary"):
                self._append_async(f"[Krogh] {plot_data['hypoxic_burden_summary']}")
            if plot_data.get("radius_sensitivity_summary"):
                self._append_async(f"[Krogh] {plot_data['radius_sensitivity_summary']}")
            uncertainty = plot_data.get("uncertainty", {})
            uncertainty_summary = uncertainty.get("summary")
            if uncertainty_summary:
                self._append_async(f"[Krogh] uncertainty band: {uncertainty_summary}")
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

        fig = plt.figure(figsize=(12.8, 7.8))

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
        ax3d.set_xlabel(self.t("xlabel_radial_position"), labelpad=8)
        ax3d.set_ylabel(self.t("ylabel_relative_length"), labelpad=10)
        ax3d.set_zlabel(self.t("zlabel_po2"), labelpad=6)
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
        compact_labels = {
            "normoxia": "Normoxia",
            "mild_hypoxia": "Mild hypoxia",
            "compensated_hypoxia": "Compensated",
            "severe_hypoxia": "Severe",
            "profound_hypoxia": "Profound",
        }
        labels = [compact_labels.get(s, self._format_oxygenation_state_label(s)) for s in states_ordered]
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
        ax_bar.tick_params(axis="y", labelsize=8, pad=2)
        ax_bar.margins(y=0.10)
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
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.90, wspace=0.34)

        if plot_data.get("fit_warning"):
            fig.text(
                0.5, 0.975,
                "Best joint compromise between tissue sensor PO2 and venous saturation",
                ha="center", va="top", fontsize=8, color=alert_colour,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
            )

        venous_label = "venous hint" if float(plot_data.get("venous_weight", 0.0)) < 0.3 else "venous target"
        uncertainty = plot_data.get("uncertainty", {})
        uncertainty_suffix = ""
        if uncertainty:
            uncertainty_suffix = (
                f" | mitoP50 range={float(uncertainty.get('p_half_low', plot_data['P_half_fit'])):.3f}"
                f"–{float(uncertainty.get('p_half_high', plot_data['P_half_fit'])):.3f} mmHg"
            )

        # Bottom status bar with alert colour
        fig.text(
            0.5, 0.01,
            f"sensor target: {plot_data['sensor_target']:.1f} mmHg | sensor avg: {plot_data['sensor_avg']:.1f} mmHg | "
            f"{venous_label}: {plot_data['P_v_target']:.1f} mmHg | simulated P_venous: {plot_data['P_v_sim']:.1f} mmHg | "
            f"perfusion={plot_data['perfusion_factor']:.2f}x | fitted mitoP50 (P\u00bd): {plot_data['P_half_fit']:.3f} mmHg"
            f"{uncertainty_suffix}",
            ha="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=alert_colour, alpha=0.20),
        )

        report_figure_path = os.path.join(CURRENT_DIR, "krogh_figures", "latest_krogh_reconstruction_report.png")
        try:
            os.makedirs(os.path.dirname(report_figure_path), exist_ok=True)
            fig.savefig(report_figure_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
            if isinstance(getattr(self, "last_krogh_reconstruction", None), dict):
                self.last_krogh_reconstruction["report_figure_path"] = report_figure_path
        except Exception:
            pass

        try:
            fig.canvas.manager.set_window_title(
                f"Krogh Reconstruction — {state_label} ({alert})"
            )
        except Exception:
            pass

        plt.show(block=False)

    def _clear(self):
        self.ui_execution.clear_output(self)

    def _quit_application(self):
        self.ui_execution.quit_application(self)

    def _run(self):
        self.ui_execution.run(self)

    def _run_default_script(self):
        self.ui_execution.run_default_script(self)

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

            case_definitions = self.series_runner.build_case_definitions(
                base_params,
                sweep_field_label,
                sweep_values,
                secondary_field_label=secondary_field_label if series_dimension == "2d" else None,
                secondary_values=secondary_values,
            )

            def _per_case_callback(case_definition, total_count):
                self._set_status_async(
                    self.t(
                        "status_series_case",
                        index=case_definition["case_index"],
                        count=total_count,
                    )
                )
                if series_dimension == "2d":
                    self._append_async(
                        self.t(
                            "series_case_progress_2d",
                            index=case_definition["case_index"],
                            count=total_count,
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
                            count=total_count,
                            parameter=self._field_label(sweep_field_label),
                            value=case_definition["sweep_value"],
                        )
                    )
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

            results = self.series_runner.run_case_definitions(
                case_definitions,
                numeric_settings,
                per_case_callback=_per_case_callback,
            )

            results_df = pd.DataFrame(results)
            numerics_report = self.series_runner.analyze_numerics(
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
        self.plot_workflow.show_series_plot(
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
        )

    def _show_series_surface_plots(
        self,
        results_df,
        sweep_field_label,
        secondary_field_label,
        selected_plot_fields,
        parameter_text,
        style,
    ):
        return self.plot_workflow.show_series_surface_plots(
            self,
            results_df,
            sweep_field_label,
            secondary_field_label,
            selected_plot_fields,
            parameter_text,
            style,
        )

    def _show_series_heatmaps(
        self,
        results_df,
        sweep_field_label,
        secondary_field_label,
        selected_plot_fields,
        parameter_text,
        style,
    ):
        return self.plot_workflow.show_series_heatmaps(
            self,
            results_df,
            sweep_field_label,
            secondary_field_label,
            selected_plot_fields,
            parameter_text,
            style,
        )

    def _save_series_run_bundle(self, parent_dir, figures, results_export_df, setup_df, bundle_context):
        self.ui_figures.save_series_run_bundle(self, parent_dir, figures, results_export_df, setup_df, bundle_context)

    def _format_run_bundle_parameters(self, bundle_context):
        return self.ui_figures.format_run_bundle_parameters(bundle_context)

    def _offer_figure_display(self):
        self.ui_figures.offer_figure_display(self)

    def _show_figure_dialog(self, fig_names, figures_dir):
        self.ui_figures.show_figure_dialog(self, fig_names, figures_dir)

    def _display_figures(self, fig_names, figures_dir):
        self.ui_figures.display_figures(fig_names, figures_dir)

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
        self.plot_workflow.show_3d_plot(self, plot_data)

    def _save_case(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[(self.t("json_files"), "*.json"), (self.t("all_files"), "*")],
            title=self.t("save_case_title"),
        )
        if not path:
            return
        try:
            data = self.case_repository.build_case_payload(self)
            self.case_repository.save_to_path(data, path)
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
            data = self.case_repository.load_from_path(path)
            self.case_repository.apply_case_payload(self, data)
        except Exception as exc:
            messagebox.showerror(self.t("load_error_title"), str(exc))
            return

        self._append(self.t("case_loaded_from", path=path))


if __name__ == "__main__":
    app = KroghGUI()
    app.mainloop()