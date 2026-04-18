"""Figure-display and series-bundle helpers for the Krogh GUI application."""

from __future__ import annotations

import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd


class UIFigureCoordinator:
    """Handles saved-figure dialogs and series result bundle export."""

    def __init__(self, *, project_dir: str):
        self.project_dir = project_dir

    def save_series_run_bundle(self, gui, parent_dir, figures, results_export_df, setup_df, bundle_context):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(parent_dir, f"krogh_series_run_{timestamp}")
        suffix = 1
        while os.path.exists(run_dir):
            suffix += 1
            run_dir = os.path.join(parent_dir, f"krogh_series_run_{timestamp}_{suffix}")
        os.makedirs(run_dir, exist_ok=False)

        excel_path = os.path.join(run_dir, "series_results.xlsx")
        results_path = excel_path
        setup_path = None
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                results_export_df.to_excel(writer, sheet_name=gui.t("sheet_series_results"), index=False)
                setup_df.to_excel(writer, sheet_name=gui.t("sheet_series_setup"), index=False)
        except ModuleNotFoundError:
            results_path = os.path.join(run_dir, "series_results.csv")
            setup_path = os.path.join(run_dir, "series_setup.csv")
            results_export_df.to_csv(results_path, index=False)
            setup_df.to_csv(setup_path, index=False)
            gui._append("openpyxl not available; saved CSV files instead of an Excel workbook.")

        style = gui._get_series_plot_style(
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
            gui._append(gui.t("plot_saved", path=plot_file))

        params_path = os.path.join(run_dir, "run_parameters.txt")
        with open(params_path, "w", encoding="utf-8") as handle:
            handle.write(self.format_run_bundle_parameters(bundle_context))

        gui._append(gui.t("results_saved", path=results_path))
        if setup_path is not None:
            gui._append(f"Series setup saved to: {setup_path}")
        gui._append(gui.t("bundle_file_parameters", path=params_path))
        gui._append(gui.t("bundle_saved", path=run_dir))

    def format_run_bundle_parameters(self, bundle_context):
        lines = []
        lines.append("Krogh GUI series run bundle")
        lines.append(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
        lines.append("mode=series")
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

    def offer_figure_display(self, gui):
        figures_dir = os.path.join(self.project_dir, "krogh_figures")
        if not os.path.isdir(figures_dir):
            return
        png_files = sorted(f for f in os.listdir(figures_dir) if f.endswith("_highres.png"))
        if not png_files:
            return
        fig_names = [f[: -len("_highres.png")] for f in png_files]
        self.show_figure_dialog(gui, fig_names, figures_dir)

    def show_figure_dialog(self, gui, fig_names, figures_dir):
        dlg = tk.Toplevel(gui)
        dlg.title(gui.t("show_figures_title"))
        dlg.geometry("520x560")
        dlg.minsize(400, 320)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text=gui.t("select_figures"),
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
                self.display_figures(selected, figures_dir)

        ttk.Button(btn_frame, text=gui.t("select_all"), command=select_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=gui.t("deselect_all"), command=deselect_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=gui.t("show_selected"), command=show_selected).pack(side="left", padx=4)
        ttk.Button(btn_frame, text=gui.t("cancel"), command=dlg.destroy).pack(side="right", padx=4)

    def display_figures(self, fig_names, figures_dir):
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


__all__ = ["UIFigureCoordinator"]
