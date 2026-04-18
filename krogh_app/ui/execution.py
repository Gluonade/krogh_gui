"""Execution-flow helpers for the Krogh GUI application."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk

import matplotlib.pyplot as plt


class UIExecutionCoordinator:
    """Coordinates run dispatching and application-level execution actions."""

    def __init__(self, *, project_dir: str):
        self.project_dir = project_dir

    def clear_output(self, gui) -> None:
        gui.output.delete("1.0", "end")
        gui._set_status(gui.t("status_ready"))

    def quit_application(self, gui) -> None:
        if getattr(gui, "_is_closing", False):
            return
        gui._is_closing = True
        for tooltip in getattr(gui, "numeric_tooltips", []):
            tooltip.destroy()
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            gui.quit()
        except (RuntimeError, tk.TclError):
            pass
        try:
            gui.destroy()
        except (RuntimeError, tk.TclError):
            pass

    def run(self, gui) -> None:
        gui._set_progress_running(True)
        mode = gui.mode_var.get()

        if mode == "default":
            gui._set_status(gui.t("status_running_default"))
            threading.Thread(target=gui._run_default_script, daemon=True).start()
            return

        if mode == "series":
            gui._set_status(gui.t("status_running_series"))
            series_params = gui._get_series_inputs()
            if series_params is None:
                gui._set_progress_running(False)
                gui._set_status(gui.t("status_ready"))
                return

            threading.Thread(
                target=gui._run_series_worker,
                kwargs=series_params,
                daemon=True,
            ).start()
            return

        params = gui._get_single_case_inputs()
        if params is None:
            gui._set_progress_running(False)
            gui._set_status(gui.t("status_ready"))
            return

        numeric_settings = gui._get_numeric_settings_inputs()
        if numeric_settings is None:
            gui._set_progress_running(False)
            gui._set_status(gui.t("status_ready"))
            return

        gui._set_status(gui.t("status_running_single"))
        threading.Thread(
            target=gui._run_single_case_worker,
            kwargs={
                **params,
                "numeric_settings": numeric_settings,
                "result_label_context": gui._build_result_label_context(params),
            },
            daemon=True,
        ).start()

    def run_default_script(self, gui) -> None:
        script_path = os.path.join(self.project_dir, "krogh_basis.py")
        if not os.path.exists(script_path):
            gui._append_async(gui.t("default_not_found"))
            gui._set_status_async(gui.t("status_error"))
            gui._call_on_ui_thread(gui._set_progress_running, False)
            return

        gui._append_async(gui.t("running_default"))
        success = False
        try:
            proc = subprocess.run(
                [sys.executable, script_path, "--no-show"],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True,
                check=False,
            )
            gui._append_async(gui.t("return_code", code=proc.returncode))
            if proc.stdout.strip():
                gui._append_async(gui.t("stdout_last"))
                lines = proc.stdout.splitlines()
                gui._append_async("\n".join(lines[-40:]))
            if proc.stderr.strip():
                gui._append_async(gui.t("stderr"))
                gui._append_async(proc.stderr)
            success = True
        except Exception as exc:
            gui._append_async(gui.t("default_run_error", error=exc))
            gui._set_status_async(gui.t("status_error"))
        finally:
            if success:
                gui._set_status_async(gui.t("status_finished"))
            gui._call_on_ui_thread(gui._set_progress_running, False)
            gui._call_on_ui_thread(gui._offer_figure_display)


__all__ = ["UIExecutionCoordinator"]
