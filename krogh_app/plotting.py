"""Shared plotting helpers for the Krogh GUI project."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from tkinter import filedialog


@dataclass(frozen=True)
class PlotStyle:
    figsize_2d: tuple[float, float]
    figsize_heatmap: tuple[float, float]
    figsize_3d: tuple[float, float]
    annotation_fontsize: int
    legend_fontsize: int
    wrap_width: int
    axes_top: float
    save_dpi: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "figsize_2d": self.figsize_2d,
            "figsize_heatmap": self.figsize_heatmap,
            "figsize_3d": self.figsize_3d,
            "annotation_fontsize": self.annotation_fontsize,
            "legend_fontsize": self.legend_fontsize,
            "wrap_width": self.wrap_width,
            "axes_top": self.axes_top,
            "save_dpi": self.save_dpi,
        }


class PlotManager:
    @staticmethod
    def wrap_annotation(text: Any, width: int = 118) -> str:
        wrapped_lines: list[str] = []
        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
            )
        return "\n".join(wrapped_lines)

    @staticmethod
    def get_series_plot_style(publication_mode: bool = False, publication_layout: str = "wide") -> dict[str, Any]:
        if publication_mode:
            if publication_layout == "a4":
                return PlotStyle(
                    figsize_2d=(11.7, 8.3),
                    figsize_heatmap=(11.7, 8.3),
                    figsize_3d=(11.7, 8.6),
                    annotation_fontsize=9,
                    legend_fontsize=9,
                    wrap_width=96,
                    axes_top=0.68,
                    save_dpi=300,
                ).to_dict()
            return PlotStyle(
                figsize_2d=(14.2, 8.0),
                figsize_heatmap=(14.2, 8.0),
                figsize_3d=(14.2, 8.4),
                annotation_fontsize=9,
                legend_fontsize=9,
                wrap_width=108,
                axes_top=0.70,
                save_dpi=300,
            ).to_dict()
        return PlotStyle(
            figsize_2d=(13.0, 8.4),
            figsize_heatmap=(13.0, 8.2),
            figsize_3d=(13.0, 8.8),
            annotation_fontsize=8,
            legend_fontsize=8,
            wrap_width=118,
            axes_top=0.70,
            save_dpi=200,
        ).to_dict()


class PlotWorkflowCoordinator:
    """Handles plot rendering while reusing the GUI for labels and messages."""

    def __init__(self, *, hypoxic_fields, r_cap: float, r_tis: float):
        self.hypoxic_fields = tuple(hypoxic_fields)
        self.r_cap = r_cap
        self.r_tis = r_tis

    def show_series_plot(
        self,
        gui,
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
        x_label = gui._field_label(sweep_field_label)
        style = gui._get_series_plot_style(publication_mode, publication_layout)
        parameter_text = gui._wrap_plot_annotation(
            gui._format_series_plot_parameters(results_df, sweep_field_label, secondary_field_label),
            width=style["wrap_width"],
        )
        figures = []
        hypoxic_fields_in_selection = [field for field in selected_plot_fields if field in self.hypoxic_fields]
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
            figures = self.show_series_surface_plots(
                gui,
                results_df,
                sweep_field_label,
                secondary_field_label,
                selected_plot_fields,
                parameter_text,
                style,
            )
        elif secondary_field_label and series_plot_mode == "heatmap":
            figures = self.show_series_heatmaps(
                gui,
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
                            label=gui.t(
                                "series_curve_legend",
                                field=gui._field_label(secondary_field_label),
                                value=float(secondary_value),
                            ),
                        )
                else:
                    y_values = results_df[field_name].to_numpy()
                    ax.plot(x_values, y_values, marker="o", linewidth=2.0)
                ax.set_ylabel(gui._result_label(field_name), labelpad=10)
                ax.set_title(gui._result_label(field_name), pad=10)
                ax.tick_params(axis="y", pad=6)
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(x_label)
                if shared_hypoxic_ylim is not None and field_name in self.hypoxic_fields:
                    ax.set_ylim(*shared_hypoxic_ylim)
                if secondary_field_label:
                    ax.legend(loc="best", fontsize=style["legend_fontsize"], ncol=2)
                fig.suptitle(gui.t("series_plot_title"), y=0.98)
                fig.text(0.5, 0.93, parameter_text, ha="center", va="top", fontsize=style["annotation_fontsize"], linespacing=1.18)
                fig.text(
                    0.5,
                    0.84,
                    gui._wrap_plot_annotation(
                        gui.t("series_plot_explanation", description=gui._result_description(field_name)),
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
                        gui.t("series_plot_window_field", field=gui._result_label(field_name))
                    )
                except Exception:
                    pass
                figures.append((field_name, fig))

        plt.show(block=True)

        if save_bundle_after_display:
            selected_dir = filedialog.askdirectory(title=gui.t("save_bundle_title"))
            if selected_dir:
                gui._save_series_run_bundle(
                    selected_dir,
                    figures,
                    results_export_df,
                    setup_df,
                    bundle_context,
                )
            else:
                gui._append(gui.t("bundle_save_cancelled"))
        else:
            gui._append(gui.t("series_results_not_saved"))
            gui._append(gui.t("series_plots_not_saved"))

        gui._set_status(gui.t("status_finished"))

    def show_series_surface_plots(
        self,
        gui,
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
            ax.set_xlabel(gui._field_label(sweep_field_label), labelpad=10)
            ax.set_ylabel(gui._field_label(secondary_field_label), labelpad=10)
            ax.set_zlabel(gui._result_label(field_name), labelpad=10)
            ax.set_title(gui._result_label(field_name), pad=14)
            fig.suptitle(gui.t("series_plot_title"), y=0.98)
            fig.text(
                0.5,
                0.93,
                gui._wrap_plot_annotation(parameter_text, width=style["wrap_width"]),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.18,
            )
            fig.text(
                0.5,
                0.84,
                gui._wrap_plot_annotation(
                    gui.t("series_plot_explanation", description=gui._result_description(field_name)),
                    width=style["wrap_width"],
                ),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.15,
                wrap=True,
            )
            fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, label=gui._result_label(field_name))
            fig.subplots_adjust(left=0.02, right=0.96, bottom=0.08, top=style["axes_top"] + 0.02)

            try:
                fig.canvas.manager.set_window_title(
                    gui.t("series_plot_surface_title", field=gui._result_label(field_name))
                )
            except Exception:
                pass
            figures.append((field_name, fig))

        return figures

    def show_series_heatmaps(
        self,
        gui,
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
            ax.set_xlabel(gui._field_label(sweep_field_label))
            ax.set_ylabel(gui._field_label(secondary_field_label))
            ax.set_title(gui._result_label(field_name), pad=10)
            fig.colorbar(image, ax=ax, label=gui._result_label(field_name))
            fig.suptitle(gui.t("series_plot_title"), y=0.98)
            fig.text(
                0.5,
                0.93,
                gui._wrap_plot_annotation(parameter_text, width=style["wrap_width"]),
                ha="center",
                va="top",
                fontsize=style["annotation_fontsize"],
                linespacing=1.18,
            )
            fig.text(
                0.5,
                0.84,
                gui._wrap_plot_annotation(
                    gui.t("series_plot_explanation", description=gui._result_description(field_name)),
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
                    gui.t("series_plot_heatmap_title", field=gui._result_label(field_name))
                )
            except Exception:
                pass
            figures.append((field_name, fig))

        return figures

    def show_3d_plot(self, gui, plot_data):
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

        r_cap_um = self.r_cap * 1e4
        z_rel_vec = plot_data["Z_rel"][:, 0]
        n = len(z_rel_vec)
        for sign in (+1, -1):
            x_curt = np.full((2, n), sign * r_cap_um)
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
            ax3d.plot([sign * r_cap_um] * n, z_rel_vec, plot_data["P_c_axial"], "k-", lw=1.5)

        ax3d.plot(
            np.zeros_like(z_rel_vec),
            z_rel_vec,
            plot_data["P_avg"],
            "r-",
            lw=3.0,
            label=gui.t("legend_sensor_avg"),
            zorder=15,
        )

        ax3d.set_xlabel(gui.t("xlabel_radial_position"))
        ax3d.set_ylabel(gui.t("ylabel_relative_length"))
        ax3d.set_zlabel(gui.t("zlabel_po2"))
        ax3d.set_xlim(-self.r_tis * 1e4, self.r_tis * 1e4)
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
            gui.t(
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
        fig.colorbar(surf, cax=cbar_ax, orientation="horizontal", label=gui.t("colorbar_po2"))
        fig.subplots_adjust(bottom=0.10)
        plt.show(block=False)


__all__ = ["PlotManager", "PlotStyle", "PlotWorkflowCoordinator"]
