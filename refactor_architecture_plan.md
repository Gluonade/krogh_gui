# Refactoring plan for the Krogh GUI project

## Goal
The current application works and is scientifically useful, but too much responsibility is concentrated in one large file and one large GUI class. The next development step should therefore be a safe, incremental refactoring that improves readability, maintainability, and extensibility without changing the validated mathematical behavior.

## Guiding principle
Use a hybrid architecture:
- keep the low-level mathematical and numerical kernel mostly functional,
- move configuration, workflow orchestration, plotting, persistence, and GUI state handling into focused classes.

This avoids over-engineering while still making the project much easier to extend.

---

## Recommended target package structure

```text
Tkinter_GUI/
├─ app.py                        # small startup entry point
├─ krogh_app/
│  ├─ __init__.py
│  ├─ constants.py               # physical constants and defaults
│  ├─ types.py                   # dataclasses for inputs/results/settings
│  ├─ numerics.py                # low-level physiological and solver functions
│  ├─ model.py                   # KroghModel class wrapping the simulation workflow
│  ├─ diagnostics.py             # DiagnosticEngine class
│  ├─ reconstruction.py          # KroghReconstructor class
│  ├─ series.py                  # SeriesRunner class for 1D/2D sweeps
│  ├─ plotting.py                # PlotManager class
│  ├─ persistence.py             # CaseRepository and BundleExporter classes
│  ├─ localization.py            # TranslationManager class
│  └─ ui/
│     ├─ __init__.py
│     ├─ main_window.py          # Main application window
│     ├─ tooltips.py             # ToolTip class
│     ├─ input_panel.py          # single-case and general inputs
│     ├─ series_panel.py         # series controls
│     ├─ numerics_panel.py       # solver settings UI
│     ├─ diagnostic_panel.py     # diagnostic and reconstruction UI
│     └─ dialogs.py              # help windows and file dialogs
├─ krogh_GUI.py                  # legacy compatibility wrapper during transition
├─ oxygenation_diagnostic_mvp.py # can remain temporarily, then be absorbed or wrapped
└─ refactor_architecture_plan.md
```

---

## Concrete classes to introduce

### 1. Simulation dataclasses
These should be plain data containers.

#### SingleCaseInput
Fields:
- inlet oxygen pressure
- mitochondrial half-saturation parameter
- pH
- pCO2
- temperature
- perfusion factor
- axial diffusion flag
- thresholds for summary metrics

#### NumericSettings
Fields:
- ODE relative tolerance
- ODE absolute tolerance
- maximum ODE step
- axial diffusion iteration limit
- axial diffusion tolerance
- axial coupling iteration limit
- axial coupling tolerance

#### DiagnosticInput
Fields:
- blood-gas oxygen
- pCO2
- pH
- temperature
- sensor oxygen
- optional hemoglobin
- optional venous saturation
- alert thresholds

#### SimulationResult
Fields:
- effective P50
- venous PO2
- tissue minimum
- percentile values
- hypoxic fractions
- sensor average
- saturation estimates
- full capillary and tissue arrays for plotting

---

### 2. KroghModel
Main responsibility: run the mechanistic oxygen transport model.

Methods:
- effective_p50(...)
- hill_saturation(...)
- dC_dP(...)
- krogh_erlang(...)
- michaelis_menten_consumption(...)
- solve_initial_capillary_po2(...)
- solve_tissue_field(...)
- solve_coupled_case(...)
- run_single_case(...)

Opinion:
This should be the scientific core class. It may internally call mostly pure helper functions, but it gives the rest of the application one stable API.

---

### 3. DiagnosticEngine
Main responsibility: probabilistic interpretation of blood-gas and sensor values.

Methods:
- evaluate(input_data)
- compute_feature_risks(...)
- compute_state_probabilities(...)
- make_alert_decision(...)

The current logic from the diagnostic MVP fits very naturally here.

---

### 4. KroghReconstructor
Main responsibility: inverse fitting from diagnostic values back to plausible model parameters.

Methods:
- venous_saturation_to_po2(...)
- fit_p_half_from_venous(...)
- fit_joint_parameters(...)
- build_reconstruction_summary(...)

This class should depend on KroghModel instead of containing duplicated simulation logic.

---

### 5. SeriesRunner
Main responsibility: 1D and 2D parameter sweeps.

Methods:
- build_series_values(...)
- run_1d_series(...)
- run_2d_series(...)
- analyze_series_numerics(...)

This removes loop-heavy sweep code from the GUI class and makes it much easier to test.

---

### 6. PlotManager
Main responsibility: all plotting and figure export.

Methods:
- show_single_case_plot(...)
- show_3d_plot(...)
- show_series_plot(...)
- show_surface_plots(...)
- show_heatmaps(...)
- save_figures(...)

This is one of the most important extractions because plotting code usually makes GUI classes much harder to navigate.

---

### 7. CaseRepository
Main responsibility: save and load cases.

Methods:
- save_case(...)
- load_case(...)
- capture_ui_state(...)
- restore_ui_state(...)

---

### 8. BundleExporter
Main responsibility: export tables, figures, and run metadata.

Methods:
- save_series_bundle(...)
- format_bundle_parameters(...)
- save_excel_results(...)

---

### 9. TranslationManager
Main responsibility: language lookup and formatting.

Methods:
- translate(key, **kwargs)
- field_label(...)
- result_label(...)
- numeric_label(...)

This will make the main GUI class smaller immediately.

---

### 10. MainWindow
Main responsibility: user interaction only.

It should coordinate widgets, read inputs, call services, and display results, but it should no longer contain most of the numerical logic itself.

A good rule is:
- UI class handles events,
- service classes perform work,
- data classes carry data,
- plotting and persistence classes handle output.

---

## Mapping from the current file to the new structure

### Move out of the current GUI class first
Highest-priority extractions:
1. plotting methods
2. save/load and export methods
3. diagnostic + reconstruction methods
4. numerics help and translation helpers
5. sweep execution workers

### Keep functional for now
These can remain plain functions inside a model or numerics module:
- effective P50 calculation
- Hill saturation
- differential capacity
- Krogh radial formula
- Michaelis-Menten consumption
- finite-difference update helpers

This is the safest balance between clarity and scientific reproducibility.

---

## Recommended migration order

### Phase 1: safe non-invasive extraction
- create the new package folder
- move dataclasses and constants first
- move translation dictionary and tooltip helper
- keep old imports working

### Phase 2: separate the scientific core
- extract numerical functions into a dedicated model/numerics module
- introduce KroghModel as a wrapper API
- verify that single-case outputs are unchanged

### Phase 3: separate diagnostics and reconstruction
- move diagnostic logic into DiagnosticEngine
- move inverse fitting into KroghReconstructor
- verify diagnostic outputs with known examples

### Phase 4: separate plotting and export
- create PlotManager and BundleExporter
- remove plotting-heavy methods from the main GUI class

### Phase 5: simplify the GUI shell
- reduce the current giant GUI file to a smaller entry and controller layer
- keep the old file temporarily as a compatibility launcher

---

## What should not be done immediately
To keep risk low, the following should not be the first step:
- a full rewrite into deeply nested inheritance hierarchies
- changing the validated numerical equations during restructuring
- mixing GUI changes and mathematical changes in the same refactor pass

The safest approach is incremental extraction with repeated verification.

---

## Practical final recommendation
For this project, the best next implementation step is not a complete rewrite but this first real milestone:

1. create a small package folder,
2. introduce dataclasses,
3. extract KroghModel, DiagnosticEngine, PlotManager, and CaseRepository,
4. keep the existing GUI visually unchanged.

That would already give a major improvement in code structure while preserving the current scientific behavior and user workflow.