"""Main window layout builder for the Krogh GUI application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from krogh_app.constants import NUMERIC_SETTINGS_FIELDS
from krogh_app.ui.tooltips import ToolTip


class UIWindowBuilder:
    """Builds the Tkinter window layout while keeping the main GUI class slimmer."""

    def __init__(self, *, language_names, series_sweep_fields, series_result_fields, get_numeric_settings):
        self.language_names = language_names
        self.series_sweep_fields = series_sweep_fields
        self.series_result_fields = series_result_fields
        self.get_numeric_settings = get_numeric_settings

    def build(self, gui) -> None:
        gui.title(gui.t("app_title"))

        main_frame = ttk.Frame(gui)
        main_frame.pack(fill="both", expand=True)

        self._build_language_frame(gui, main_frame)
        self._build_mode_frame(gui, main_frame)
        self._build_settings_notebook(gui, main_frame)
        self._build_controls(gui, main_frame)
        self._build_progress_frame(gui, main_frame)
        self._build_output_frame(gui, main_frame)

        gui._toggle_inputs()

    def _build_language_frame(self, gui, main_frame):
        language_frame = ttk.Frame(main_frame)
        language_frame.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(language_frame, text=gui.t("language_label")).pack(side="left", padx=(0, 8))
        gui.language_name_to_code = {name: code for code, name in self.language_names.items()}
        gui.language_code_to_name = {code: name for code, name in self.language_names.items()}
        gui.language_combo = ttk.Combobox(
            language_frame,
            textvariable=gui.language_display_var,
            values=[gui.language_code_to_name[code] for code in self.language_names],
            state="readonly",
            width=14,
        )
        gui.language_combo.pack(side="left")
        gui.language_display_var.set(gui.language_code_to_name[gui.language_code])
        gui.language_combo.bind("<<ComboboxSelected>>", gui._on_language_selected)

    def _build_mode_frame(self, gui, main_frame):
        top = ttk.LabelFrame(main_frame, text=gui.t("mode_group"))
        top.pack(fill="x", padx=10, pady=8)

        ttk.Radiobutton(
            top,
            text=gui.t("mode_default"),
            variable=gui.mode_var,
            value="default",
            command=gui._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Radiobutton(
            top,
            text=gui.t("mode_single"),
            variable=gui.mode_var,
            value="single",
            command=gui._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Radiobutton(
            top,
            text=gui.t("mode_series"),
            variable=gui.mode_var,
            value="series",
            command=gui._toggle_inputs,
        ).pack(anchor="w", padx=8, pady=4)

    def _build_settings_notebook(self, gui, main_frame):
        gui.settings_notebook = ttk.Notebook(main_frame)
        gui.settings_notebook.pack(fill="x", padx=10, pady=(0, 8))

        self._build_params_tab(gui)
        self._build_series_tab(gui)
        self._build_numerics_tab(gui)
        self._build_diagnostic_tab(gui)

    def _build_params_tab(self, gui):
        params = ttk.Frame(gui.settings_notebook)
        gui.params_tab = params
        gui.settings_notebook.add(params, text=gui.t("tab_inputs"))

        gui.entries = {}
        fields = [
            ("PO2_inlet_mmHg", "100.0"),
            ("mitoP50_mmHg", "1.0"),
            ("pH", "7.40"),
            ("pCO2_mmHg", "40.0"),
            ("Temp_C", "37.0"),
            ("Perfusion_factor", "1.0"),
            ("Metabolic_rate_rel", "1.0"),
            ("Tissue_radius_um", "100"),
            ("High_PO2_threshold_1_mmHg", "100.0"),
            ("High_PO2_threshold_2_mmHg", "200.0"),
            ("High_PO2_additional_thresholds_mmHg", ""),
            ("High_PO2_relative_thresholds_percent", "90,50,30"),
            ("Relative_PO2_reference", "inlet"),
        ]

        for i, (name, default) in enumerate(fields):
            r = i // 3
            c = (i % 3) * 2
            ttk.Label(params, text=gui._field_label(name)).grid(row=r, column=c, padx=8, pady=6, sticky="e")
            if name == "Relative_PO2_reference":
                ref_values = ("inlet", "tissue_max")
                ref_labels = tuple(gui.t(f"reference_{v}") for v in ref_values)
                entry = ttk.Combobox(
                    params,
                    values=ref_labels,
                    state="readonly",
                    width=16,
                )
                entry.set(gui.t(f"reference_{default}"))
            elif name == "Tissue_radius_um":
                entry = ttk.Combobox(
                    params,
                    values=("30", "50", "100"),
                    state="readonly",
                    width=12,
                )
                entry.set(str(default))
            else:
                entry_width = 28 if name in {"High_PO2_additional_thresholds_mmHg", "High_PO2_relative_thresholds_percent"} else 12
                entry = ttk.Entry(params, width=entry_width)
                entry.insert(0, default)
            entry.grid(row=r, column=c + 1, padx=8, pady=6, sticky="w")
            gui.entries[name] = entry

        ttk.Checkbutton(
            params,
            text=gui.t("include_axial"),
            variable=gui.include_axial_var,
        ).grid(row=4, column=0, columnspan=3, padx=8, pady=6, sticky="w")

    def _build_series_tab(self, gui):
        series_frame = ttk.Frame(gui.settings_notebook)
        gui.series_tab = series_frame
        gui.settings_notebook.add(series_frame, text=gui.t("tab_series"))

        ttk.Label(series_frame, text=gui.t("varying_parameter")).grid(row=0, column=0, padx=8, pady=6, sticky="e")
        ttk.Label(series_frame, text=gui.t("series_dimension")).grid(row=0, column=2, padx=8, pady=6, sticky="e")
        gui.series_param_key_to_display = {
            field_key: gui._field_label(field_key) for field_key in self.series_sweep_fields
        }
        gui.series_param_display_to_key = {
            label: key for key, label in gui.series_param_key_to_display.items()
        }
        gui.series_param_combo = ttk.Combobox(
            series_frame,
            textvariable=gui.series_param_var,
            values=list(gui.series_param_key_to_display.values()),
            state="readonly",
            width=18,
        )
        gui.series_param_combo.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        gui._set_series_param_display(gui.series_param_key)
        gui.series_dimension_combo = ttk.Combobox(
            series_frame,
            textvariable=gui.series_dimension_var,
            values=("1d", "2d"),
            state="readonly",
            width=10,
        )
        gui.series_dimension_combo.grid(row=0, column=3, padx=8, pady=6, sticky="w")
        gui.series_dimension_combo.bind("<<ComboboxSelected>>", lambda _event: gui._toggle_series_dimension_inputs())

        gui.series_entries = {}
        gui.series_entries_by_key = {}
        series_fields = [
            ("start_value", gui.t("start_value"), "80.0"),
            ("end_value", gui.t("end_value"), "120.0"),
            ("step_size", gui.t("step_size"), "5.0"),
            ("secondary_start_value", gui.t("secondary_start_value"), "0.5"),
            ("secondary_end_value", gui.t("secondary_end_value"), "1.5"),
            ("secondary_step_size", gui.t("secondary_step_size"), "0.25"),
        ]
        for i, (entry_key, name, default) in enumerate(series_fields):
            row = 1 + i // 3
            column = (i % 3) * 2
            ttk.Label(series_frame, text=name).grid(row=row, column=column, padx=8, pady=6, sticky="e")
            entry_width = 16 if entry_key in {"start_value", "secondary_start_value"} else 12
            entry = ttk.Entry(series_frame, width=entry_width)
            entry.insert(0, default)
            entry.grid(row=row, column=column + 1, padx=8, pady=6, sticky="w")
            gui.series_entries[name] = entry
            gui.series_entries_by_key[entry_key] = entry

        ttk.Label(series_frame, text=gui.t("secondary_parameter")).grid(row=3, column=0, padx=8, pady=6, sticky="e")
        gui.series_param2_combo = ttk.Combobox(
            series_frame,
            textvariable=gui.series_param2_var,
            values=list(gui.series_param_key_to_display.values()),
            state="readonly",
            width=18,
        )
        gui.series_param2_combo.grid(row=3, column=1, padx=8, pady=6, sticky="w")
        gui._set_series_param2_display(gui.series_param2_key)

        ttk.Label(series_frame, text=gui.t("series_plot_mode")).grid(row=3, column=2, padx=8, pady=6, sticky="e")
        gui.series_plot_mode_combo = ttk.Combobox(
            series_frame,
            textvariable=gui.series_plot_mode_var,
            values=("2d", "3d", "heatmap"),
            state="readonly",
            width=18,
        )
        gui.series_plot_mode_combo.grid(row=3, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(series_frame, text=gui.t("plot_outputs")).grid(row=4, column=0, padx=8, pady=(8, 6), sticky="ne")
        plot_list_frame = ttk.Frame(series_frame)
        plot_list_frame.grid(row=4, column=1, columnspan=5, padx=8, pady=(8, 6), sticky="we")

        gui.series_plot_listbox = tk.Listbox(
            plot_list_frame,
            selectmode="extended",
            exportselection=False,
            height=6,
            width=32,
        )
        for field_name in self.series_result_fields:
            gui.series_plot_listbox.insert("end", gui._result_label(field_name))
        for index in (1, 2, 3, 4, 6):
            gui.series_plot_listbox.selection_set(index)
        gui.series_plot_listbox.pack(side="left", fill="x", expand=True)

        plot_scrollbar = ttk.Scrollbar(plot_list_frame, orient="vertical", command=gui.series_plot_listbox.yview)
        plot_scrollbar.pack(side="right", fill="y")
        gui.series_plot_listbox.config(yscrollcommand=plot_scrollbar.set)

        ttk.Label(
            series_frame,
            text=gui.t("multi_select_hint"),
        ).grid(row=5, column=1, columnspan=3, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(
            series_frame,
            text=gui.t("series_selection_hint"),
        ).grid(row=6, column=1, columnspan=4, padx=8, pady=(0, 4), sticky="w")
        gui.series_save_results_checkbutton = ttk.Checkbutton(
            series_frame,
            text=gui.t("save_series_results"),
            variable=gui.save_series_results_var,
        )
        gui.series_save_results_checkbutton.grid(row=7, column=0, columnspan=3, padx=8, pady=(2, 6), sticky="w")
        gui.series_lock_hypoxic_scale_checkbutton = ttk.Checkbutton(
            series_frame,
            text=gui.t("lock_hypoxic_fraction_scale"),
            variable=gui.lock_hypoxic_fraction_scale_var,
        )
        gui.series_lock_hypoxic_scale_checkbutton.grid(row=8, column=0, columnspan=4, padx=8, pady=(0, 6), sticky="w")
        gui.series_publication_mode_checkbutton = ttk.Checkbutton(
            series_frame,
            text=gui.t("publication_mode"),
            variable=gui.publication_mode_var,
            command=gui._toggle_series_dimension_inputs,
        )
        gui.series_publication_mode_checkbutton.grid(row=9, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="w")
        ttk.Label(series_frame, text=gui.t("publication_layout")).grid(row=9, column=2, padx=8, pady=(0, 6), sticky="e")
        gui.publication_layout_key_to_display = {
            "a4": gui.t("publication_layout_a4"),
            "wide": gui.t("publication_layout_wide"),
        }
        gui.publication_layout_display_to_key = {
            label: key for key, label in gui.publication_layout_key_to_display.items()
        }
        gui.publication_layout_combo = ttk.Combobox(
            series_frame,
            textvariable=gui.publication_layout_var,
            values=list(gui.publication_layout_key_to_display.values()),
            state="disabled",
            width=18,
        )
        gui.publication_layout_combo.grid(row=9, column=3, padx=8, pady=(0, 6), sticky="w")
        gui.publication_layout_combo.bind("<<ComboboxSelected>>", gui._on_publication_layout_selected)
        gui._set_publication_layout_display(gui.publication_layout_key)
        ttk.Label(
            series_frame,
            text=gui.t("series_plots_separate_hint"),
            wraplength=600,
        ).grid(row=10, column=0, columnspan=6, padx=8, pady=(0, 6), sticky="w")

        gui._toggle_series_dimension_inputs()

    def _build_numerics_tab(self, gui):
        numerics_frame = ttk.Frame(gui.settings_notebook)
        gui.numerics_tab = numerics_frame
        gui.settings_notebook.add(numerics_frame, text=gui.t("tab_numerics"))

        gui.numeric_entries = {}
        gui.numeric_tooltips = []
        numeric_defaults = self.get_numeric_settings()
        for i, (key, label_key, value_type) in enumerate(NUMERIC_SETTINGS_FIELDS):
            row = i // 4
            col = (i % 4) * 2
            label = ttk.Label(numerics_frame, text=gui._numeric_label(label_key))
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
            entry.bind("<FocusOut>", lambda _event, setting_key=key: gui._sanitize_numeric_entry(setting_key), add="+")
            entry.bind("<Return>", lambda _event, setting_key=key: gui._sanitize_numeric_entry(setting_key), add="+")
            gui.numeric_entries[key] = entry
            gui.numeric_tooltips.append(ToolTip(label, lambda setting_key=key: gui._build_numeric_field_tooltip(setting_key)))
            gui.numeric_tooltips.append(ToolTip(entry, lambda setting_key=key: gui._build_numeric_field_tooltip(setting_key)))

        ttk.Button(
            numerics_frame,
            text=gui.t("numeric_help_button"),
            command=gui._show_numeric_parameter_help,
        ).grid(row=2, column=0, padx=8, pady=(2, 8), sticky="w")
        ttk.Label(
            numerics_frame,
            text=gui.t("numeric_help_hint"),
            wraplength=700,
        ).grid(row=2, column=1, columnspan=7, padx=8, pady=(2, 8), sticky="w")

    def _build_diagnostic_tab(self, gui):
        diagnostic_frame = ttk.Frame(gui.settings_notebook)
        gui.diagnostic_tab = diagnostic_frame
        gui.settings_notebook.add(diagnostic_frame, text=gui.t("tab_diagnostic"))

        diagnostic_group = ttk.LabelFrame(diagnostic_frame, text=gui.t("diagnostic_group"))
        diagnostic_group.pack(fill="x", padx=10, pady=10)
        diagnostic_group.columnconfigure(0, weight=1)

        gui.diagnostic_entries = {}
        diagnostic_fields = [
            ("po2", "diag_po2", "80.0"),
            ("pco2", "diag_pco2", "40.0"),
            ("pH", "diag_ph", "7.40"),
            ("temperature_c", "diag_temp", "37.0"),
            ("sensor_po2", "diag_sensor_po2", "25.0"),
            ("metabolic_rate_rel", "diag_metabolic_rate_rel", "1.00"),
            ("hemoglobin_g_dl", "diag_hemoglobin", ""),
            ("venous_sat_percent", "diag_venous_sat", ""),
            ("yellow_threshold", "diag_yellow_threshold", "0.40"),
            ("orange_threshold", "diag_orange_threshold", "0.60"),
            ("red_threshold", "diag_red_threshold", "0.80"),
        ]

        column_count = 3
        diagnostic_columns = []
        for column_index in range(column_count):
            column_frame = ttk.Frame(diagnostic_group)
            column_frame.grid(row=0, column=column_index, padx=8, pady=6, sticky="nsew")
            column_frame.columnconfigure(0, weight=1)
            diagnostic_group.columnconfigure(column_index, weight=1)
            diagnostic_columns.append(column_frame)

        for i, (field_key, label_key, default_value) in enumerate(diagnostic_fields):
            column_index = i % column_count
            row = (i // column_count) * 2
            parent = diagnostic_columns[column_index]
            ttk.Label(
                parent,
                text=gui.t(label_key),
                anchor="w",
                justify="left",
                wraplength=220,
            ).grid(row=row, column=0, padx=0, pady=(0, 2), sticky="w")
            entry = ttk.Entry(parent, width=14)
            entry.insert(0, default_value)
            entry.grid(row=row + 1, column=0, padx=0, pady=(0, 6), sticky="w")
            gui.diagnostic_entries[field_key] = entry

        lower_row = 1

        ttk.Label(
            diagnostic_group,
            text=gui.t("diag_optional_hint"),
            wraplength=1100,
            justify="left",
        ).grid(row=lower_row, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="w")
        lower_row += 1

        radius_controls = ttk.Frame(diagnostic_group)
        radius_controls.grid(row=lower_row, column=0, columnspan=3, padx=8, pady=6, sticky="w")

        ttk.Label(radius_controls, text="Radius mode").grid(row=0, column=0, padx=(0, 8), pady=0, sticky="w")
        gui.diagnostic_radius_mode_combo = ttk.Combobox(
            radius_controls,
            textvariable=gui.diagnostic_radius_mode_var,
            values=["all variants", "selected radius only"],
            state="readonly",
            width=18,
        )
        gui.diagnostic_radius_mode_combo.grid(row=0, column=1, padx=(0, 16), pady=0, sticky="w")
        gui.diagnostic_radius_mode_combo.bind("<<ComboboxSelected>>", gui._toggle_diagnostic_radius_variant_controls)

        ttk.Label(radius_controls, text="Selected radius").grid(row=0, column=2, padx=(0, 8), pady=0, sticky="w")
        gui.diagnostic_radius_variant_combo = ttk.Combobox(
            radius_controls,
            textvariable=gui.diagnostic_radius_variant_var,
            values=["30 µm", "50 µm", "100 µm"],
            state="readonly",
            width=12,
        )
        gui.diagnostic_radius_variant_combo.grid(row=0, column=3, padx=0, pady=0, sticky="w")
        gui._toggle_diagnostic_radius_variant_controls()
        lower_row += 1

        diagnostic_controls = ttk.Frame(diagnostic_group)
        diagnostic_controls.grid(row=lower_row, column=0, columnspan=3, padx=8, pady=(4, 8), sticky="w")
        diagnostic_controls.columnconfigure(0, weight=1)
        controls_row_top = ttk.Frame(diagnostic_controls)
        controls_row_top.grid(row=0, column=0, sticky="w")
        controls_row_bottom = ttk.Frame(diagnostic_controls)
        controls_row_bottom.grid(row=1, column=0, sticky="w", pady=(6, 0))
        gui.use_single_case_button = ttk.Button(
            controls_row_top,
            text=gui.t("use_single_case_button"),
            command=gui._fill_diagnostic_from_single_case,
        )
        gui.use_single_case_button.pack(side="left", padx=(0, 6))
        gui.run_diagnostic_button = ttk.Button(
            controls_row_top,
            text=gui.t("run_diagnostic_button"),
            command=gui._run_diagnostic_from_inputs,
        )
        gui.run_diagnostic_button.pack(side="left", padx=(0, 6))
        gui.save_diagnostic_template_button = ttk.Button(
            controls_row_top,
            text=gui.t("save_diagnostic_template_button"),
            command=gui._save_diagnostic_calibration_template,
        )
        gui.save_diagnostic_template_button.pack(side="left")

        gui.save_diagnostic_report_button = ttk.Button(
            controls_row_top,
            text=gui.t("save_diagnostic_report_button"),
            command=gui._save_diagnostic_report,
        )
        gui.save_diagnostic_report_button.pack(side="left", padx=(6, 0))

        gui.save_publication_report_button = ttk.Button(
            controls_row_bottom,
            text=gui.t("save_publication_report_button"),
            command=gui._save_publication_report,
        )
        gui.save_publication_report_button.pack(side="left", padx=(6, 0))

        gui.reconstruct_krogh_button = ttk.Button(
            controls_row_bottom,
            text=gui.t("reconstruct_krogh_button"),
            command=gui._run_reconstruct_krogh,
        )
        gui.reconstruct_krogh_button.pack(side="left", padx=(6, 0))

        gui.run_reconstruction_benchmark_button = ttk.Button(
            controls_row_bottom,
            text=gui.t("run_reconstruction_benchmark_button"),
            command=gui._run_reconstruction_benchmark_from_gui,
        )
        gui.run_reconstruction_benchmark_button.pack(side="left", padx=(6, 0))

        gui.auto_save_radius_plots_check = ttk.Checkbutton(
            controls_row_bottom,
            text="Auto-save independent 30/50/100 µm reconstructions to Diagnostic reports",
            variable=gui.auto_save_radius_plots_var,
            onvalue=True,
            offvalue=False,
        )
        gui.auto_save_radius_plots_check.pack(side="left", padx=(14, 0))

        gui.diagnostic_output = tk.Text(diagnostic_frame, height=8, wrap="word")
        gui.diagnostic_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        gui.diagnostic_output.insert("1.0", gui.t("diag_result_header") + "\n")
        gui.diagnostic_output.config(state="disabled")

    def _build_controls(self, gui, main_frame):
        controls = ttk.Frame(main_frame)
        controls.pack(fill="x", padx=10, pady=8)

        ttk.Button(controls, text=gui.t("run_button"), command=gui._run).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("plot3d_button"), command=gui._run_3d_plot).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("clear_button"), command=gui._clear).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("help_button"), command=gui._show_output_parameter_help).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("save_case_button"), command=gui._save_case).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("load_case_button"), command=gui._load_case).pack(side="left", padx=4)
        ttk.Button(controls, text=gui.t("quit_button"), command=gui._quit_application).pack(side="right", padx=4)

    def _build_progress_frame(self, gui, main_frame):
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", padx=10, pady=(0, 4))

        gui.progress = ttk.Progressbar(progress_frame, mode="indeterminate", length=400)
        gui.progress.pack(fill="x", pady=(0, 4))

        gui.status_var = tk.StringVar(value=gui.t("status_ready"))
        gui.status_label = ttk.Label(progress_frame, textvariable=gui.status_var, anchor="w")
        gui.status_label.pack(fill="x")

    def _build_output_frame(self, gui, main_frame):
        out_frame = ttk.LabelFrame(main_frame, text=gui.t("output_group"))
        out_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        output_inner = ttk.Frame(out_frame)
        output_inner.pack(fill="both", expand=True, padx=6, pady=6)

        gui.output = tk.Text(output_inner, wrap="word")
        gui.output.pack(side="left", fill="both", expand=True)

        output_scrollbar = ttk.Scrollbar(output_inner, orient="vertical", command=gui.output.yview)
        output_scrollbar.pack(side="right", fill="y")
        gui.output.config(yscrollcommand=output_scrollbar.set)


__all__ = ["UIWindowBuilder"]
