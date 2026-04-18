"""Persistence and UI-state helpers for the Krogh GUI application."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from typing import Any

from tkinter import ttk


class CaseRepository:
    """Handles case serialization and temporary UI-state capture/restore."""

    _ENGLISH_STATE_LABELS = {
        "normoxia": "normoxia",
        "mild_hypoxia": "mild tissue hypoxia",
        "compensated_hypoxia": "compensated tissue hypoxia",
        "severe_hypoxia": "severe tissue hypoxia",
        "profound_hypoxia": "profound tissue hypoxia",
    }

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def build_case_payload(self, gui) -> dict[str, Any]:
        data = {name: entry.get() for name, entry in gui.entries.items()}
        data["include_axial_diffusion"] = gui.include_axial_var.get()
        data["numeric_settings"] = {key: entry.get() for key, entry in gui.numeric_entries.items()}
        data["diagnostic_settings"] = {key: entry.get() for key, entry in gui.diagnostic_entries.items()}
        data["diagnostic_radius_mode"] = gui.diagnostic_radius_mode_var.get() if hasattr(gui, "diagnostic_radius_mode_var") else "all variants"
        data["diagnostic_radius_variant"] = gui.diagnostic_radius_variant_var.get() if hasattr(gui, "diagnostic_radius_variant_var") else "100 µm"
        data["output_text"] = gui.output.get("1.0", "end-1c") if hasattr(gui, "output") else ""
        data["diagnostic_output_text"] = (
            gui.diagnostic_output.get("1.0", "end-1c") if hasattr(gui, "diagnostic_output") else ""
        )
        data["last_diagnostic_result"] = self._json_safe(getattr(gui, "last_diagnostic_result", None))
        data["last_krogh_reconstruction"] = self._json_safe(getattr(gui, "last_krogh_reconstruction", None))
        return data

    def _english_state_label(self, raw_state: Any) -> str:
        token = str(raw_state or "unknown").strip()
        return self._ENGLISH_STATE_LABELS.get(token, token.replace("_", " "))

    def _english_alert_label(self, raw_alert: Any) -> str:
        token = str(raw_alert or "unknown").strip()
        return token.replace("_", " ")

    def _latex_escape(self, value: Any) -> str:
        text = str(value).replace("\n", " ")
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        unicode_map = {
            "≈": r"$\approx$",
            "≤": r"$\leq$",
            "≥": r"$\geq$",
            "–": "--",
            "—": "---",
            "µ": r"$\mu$",
            "°": r"$^\circ$",
        }
        for old, new in unicode_map.items():
            text = text.replace(old, new)
        return text

    def _safe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _classify_saved_radius_alert(self, scenario: dict[str, Any]) -> str:
        alert_level = str(scenario.get("alert_level", "") or "").strip().lower()
        if alert_level and alert_level != "unknown":
            return alert_level

        sensor_avg = self._safe_float(scenario.get("sensor_avg")) or 0.0
        below_1 = self._safe_float(scenario.get("fraction_below_1")) or 0.0
        below_5 = self._safe_float(scenario.get("fraction_below_5")) or 0.0
        below_10 = self._safe_float(scenario.get("fraction_below_10")) or 0.0
        below_15 = self._safe_float(scenario.get("fraction_below_15")) or 0.0

        if below_1 > 0.01 or below_5 >= 0.10 or (sensor_avg > 0.0 and sensor_avg < 15.0):
            return "red"
        if below_5 >= 0.03 or below_10 >= 0.20 or (sensor_avg > 0.0 and sensor_avg < 18.0):
            return "orange"
        if below_10 >= 0.08 or below_15 >= 0.20 or (sensor_avg > 0.0 and sensor_avg < 23.0):
            return "yellow"
        return "green"

    def _build_radius_condition_note(self, report: dict[str, Any]) -> str:
        scenario_map = report.get("reconstruction_radius_scenarios")
        mode_token = str(report.get("reconstruction_radius_mode", "") or "").strip().lower()
        selected_label = str(report.get("reconstruction_selected_radius_label", "") or "").strip()
        selected_key = str(report.get("reconstruction_selected_radius_key", "") or "").strip()

        if not isinstance(scenario_map, dict) or not scenario_map:
            if "selected" in mode_token and selected_label:
                return f"Radius-condition note: the reported alert is intended for the selected {selected_label} tissue-radius assumption."
            return "Radius-condition note: no explicit radius-sensitivity comparison is available for this case."

        ordered_keys = ["normal_30um", "increased_50um", "high_100um"]
        labels: list[tuple[str, str, str]] = []
        selected_alert = "unknown"
        for key in ordered_keys:
            scenario = scenario_map.get(key)
            if not isinstance(scenario, dict):
                continue
            radius_um = self._safe_float(scenario.get("radius_um"))
            radius_text = f"{radius_um:.0f} µm" if radius_um is not None else key
            alert_level = self._classify_saved_radius_alert(scenario)
            labels.append((key, radius_text, alert_level))
            if key == selected_key:
                selected_alert = alert_level

        if not labels:
            return "Radius-condition note: no explicit radius-sensitivity comparison is available for this case."

        label_text = "; ".join(f"{radius_text}: {alert_level}" for _, radius_text, alert_level in labels)
        if "selected" in mode_token and selected_label:
            return (
                f"Clinical context note: this alert is tied to the selected {selected_label} tissue-radius assumption "
                f"({selected_alert}) and is most relevant if that geometry is clinically plausible for the patient."
            )

        unique_alerts = {alert_level for _, _, alert_level in labels}
        if len(unique_alerts) == 1:
            only_alert = labels[0][2]
            return (
                "Clinical context note: alert interpretation is conditional on the assumed tissue radius, "
                f"but all tested variants remained {only_alert} ({label_text}), which supports a similar clinical reading across plausible geometries."
            )

        return (
            "Clinical context note: alert interpretation is conditional on the assumed tissue radius "
            f"({label_text}). In this case, the higher alert mainly appears under the larger/swollen-tissue condition and is most relevant if that geometry is clinically plausible."
        )

    def _build_radius_summary(self, report: dict[str, Any]) -> str:
        scenario_map = report.get("reconstruction_radius_scenarios")
        stored_summary = str(report.get("reconstruction_radius_sensitivity_summary", "") or "").strip()
        if not isinstance(scenario_map, dict) or not scenario_map:
            return stored_summary or "Radius sensitivity was not evaluated for this case."

        ordered_keys = ["normal_30um", "increased_50um", "high_100um"]
        parts: list[str] = []
        alert_levels: list[str] = []
        for key in ordered_keys:
            scenario = scenario_map.get(key)
            if not isinstance(scenario, dict):
                continue
            radius_um = self._safe_float(scenario.get("radius_um"))
            sensor_avg = self._safe_float(scenario.get("sensor_avg"))
            fraction_below_10 = self._safe_float(scenario.get("fraction_below_10"))
            alert_level = self._classify_saved_radius_alert(scenario)
            alert_levels.append(alert_level)
            radius_text = f"{radius_um:.0f} µm" if radius_um is not None else key
            detail = f"{radius_text} -> {alert_level}"
            if sensor_avg is not None and fraction_below_10 is not None:
                detail += f" (mean {sensor_avg:.1f} mmHg, {100.0 * fraction_below_10:.1f}% below 10 mmHg)"
            parts.append(detail)

        summary = "Radius-conditioned summary: " + "; ".join(parts) + "."
        if len(set(alert_levels)) > 1:
            summary += " The interpretation is conditional on the assumed tissue radius, and the higher alert mainly appears under the larger/swollen-tissue geometry if that condition is clinically plausible."
        else:
            summary += " The alert pattern is similar across the tested radius assumptions, supporting a stable clinical interpretation."
        return summary

    def _build_interpretation_sections(self, report: dict[str, Any]) -> dict[str, str]:
        state_label = self._english_state_label(report.get("predicted_state"))
        alert_label = self._english_alert_label(report.get("alert_level"))
        risk_score = self._safe_float(report.get("risk_score"))
        certainty = self._safe_float(report.get("certainty"))
        confidence = self._safe_float(report.get("confidence"))
        p_half = report.get("reconstruction_P_half_fit", "n/a")
        perfusion = report.get("reconstruction_perfusion_factor", "n/a")
        uncertainty_summary = report.get("reconstruction_uncertainty_summary", "not available")
        assumption_summary = str(report.get("reconstruction_assumption_summary", "") or "").strip()
        burden_summary = str(report.get("reconstruction_hypoxic_burden_summary", "") or "").strip()
        radius_summary = self._build_radius_summary(report)
        radius_condition_note = self._build_radius_condition_note(report)
        driver_summary = str(report.get("driver_summary", "") or "").strip()
        fit_warning_value = report.get("reconstruction_fit_warning", False)

        concern_map = {
            "green": "Low concern",
            "ok": "Low concern",
            "yellow": "Mild concern",
            "orange": "Moderate concern",
            "red": "High concern",
        }
        concern = concern_map.get(str(report.get("alert_level", "")).strip().lower(), "Clinical note")

        if certainty is None:
            certainty_text = "The diagnostic separation could not be quantified from the saved data."
        elif certainty >= 0.60:
            certainty_text = "The model shows a fairly distinct separation from neighbouring physiological states."
        elif certainty >= 0.30:
            certainty_text = "The classification is directionally useful, but overlap with neighbouring states remains relevant."
        else:
            certainty_text = "The classification should be interpreted cautiously because the competing state probabilities remain relatively close together."

        clinical = (
            f"{concern}: the integrated blood-gas and sensor pattern is most consistent with {state_label}. "
            f"The current alert category is {alert_label}. "
            f"Risk score: {report.get('risk_score', 'n/a')}. "
            f"Confidence: {report.get('confidence', 'n/a')}. {certainty_text}"
        )
        clinical += f" {radius_condition_note}"
        if driver_summary:
            clean_driver = driver_summary.strip()
            prefix = "Main alert-score drivers:"
            if clean_driver.startswith(prefix):
                clean_driver = clean_driver[len(prefix):].strip()
            clean_driver = clean_driver.rstrip(".")
            if clean_driver:
                clinical += f" The main features contributing to this alert were {clean_driver}."

        findings = (
            f"The Krogh-based reconstruction suggests an effective mitoP50 near {p_half} mmHg and "
            f"a perfusion factor near {perfusion} x baseline. "
            f"The practical uncertainty band is reported as: {uncertainty_summary}."
        )
        if assumption_summary:
            findings += f" {assumption_summary}"
        if burden_summary:
            findings += f" {burden_summary}"
        if radius_summary:
            findings += f" {radius_summary}"

        cautions = (
            "This report is a decision-support summary and does not replace direct clinical assessment. "
            "The uncertainty band is practical and near-optimal, not a formal statistical confidence interval."
        )
        if risk_score is not None and risk_score >= 0.8:
            cautions += " Prompt reassessment is advisable when the overall clinical picture is compatible with severe oxygenation impairment."
        if confidence is not None and confidence < 0.6:
            cautions += " The probabilistic classification is not strongly dominant and should be checked against repeat measurements."
        if bool(fit_warning_value):
            cautions += " Reconstruction note: the diagnostic inputs are only partly representable by a single Krogh cylinder, so the reported fit should be interpreted as a best joint compromise."

        return {
            "clinical": clinical,
            "findings": findings,
            "cautions": cautions,
        }

    def _build_follow_up_recommendation(self, report: dict[str, Any]) -> str:
        alert_key = str(report.get("alert_level", "") or "").strip().lower()
        risk_score = self._safe_float(report.get("risk_score"))
        confidence = self._safe_float(report.get("confidence"))
        certainty = self._safe_float(report.get("certainty"))
        burden_10 = self._safe_float(report.get("reconstruction_hypoxic_below_10"))
        burden_5 = self._safe_float(report.get("reconstruction_hypoxic_below_5"))

        recommendations: list[str] = []
        if alert_key == "red" or (risk_score is not None and risk_score >= 0.8):
            recommendations.append("Immediate clinical review is advisable.")
            recommendations.append("Repeat measurements promptly and correlate them with symptoms, perfusion markers, and other bedside findings.")
        elif alert_key == "orange" or (risk_score is not None and risk_score >= 0.6):
            recommendations.append("Short-interval reassessment is advisable.")
            recommendations.append("Repeat measurements to confirm the trend and compare them with the overall clinical picture.")
        elif alert_key == "yellow" or (risk_score is not None and risk_score >= 0.4):
            recommendations.append("Repeat measurements if symptoms persist or if additional risk factors emerge.")
        else:
            recommendations.append("Continue routine monitoring and document future trends if new symptoms appear.")

        if (confidence is not None and confidence < 0.6) or (certainty is not None and certainty < 0.25):
            recommendations.append("Because model separation is limited, repeat measurements are especially valuable.")
        if (burden_10 is not None and burden_10 >= 0.25) or (burden_5 is not None and burden_5 >= 0.10):
            recommendations.append("The fitted model implies a non-trivial hidden hypoxic burden, so short-interval reassessment is appropriate even if the average values appear partly compensated.")

        recommendations.append("Use serial trends rather than a single value in isolation whenever possible.")
        return " ".join(recommendations)

    def build_diagnostic_report(self, gui) -> dict[str, Any]:
        report: dict[str, Any] = {
            "report_language": "English",
            "po2": gui.diagnostic_entries.get("po2").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("po2") else "",
            "pco2": gui.diagnostic_entries.get("pco2").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("pco2") else "",
            "pH": gui.diagnostic_entries.get("pH").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("pH") else "",
            "temperature_c": gui.diagnostic_entries.get("temperature_c").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("temperature_c") else "",
            "sensor_po2": gui.diagnostic_entries.get("sensor_po2").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("sensor_po2") else "",
            "hemoglobin_g_dl": gui.diagnostic_entries.get("hemoglobin_g_dl").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("hemoglobin_g_dl") else "",
            "venous_sat_percent": gui.diagnostic_entries.get("venous_sat_percent").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("venous_sat_percent") else "",
            "yellow_threshold": gui.diagnostic_entries.get("yellow_threshold").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("yellow_threshold") else "",
            "orange_threshold": gui.diagnostic_entries.get("orange_threshold").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("orange_threshold") else "",
            "red_threshold": gui.diagnostic_entries.get("red_threshold").get() if hasattr(gui, "diagnostic_entries") and gui.diagnostic_entries.get("red_threshold") else "",
        }

        diagnostic_result = getattr(gui, "last_diagnostic_result", None)
        if isinstance(diagnostic_result, dict):
            for key, value in diagnostic_result.items():
                report[str(key)] = self._json_safe(value)

        reconstruction = getattr(gui, "last_krogh_reconstruction", None)
        if isinstance(reconstruction, dict):
            for key, value in reconstruction.items():
                if key == "uncertainty" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        report[f"reconstruction_uncertainty_{sub_key}"] = self._json_safe(sub_value)
                elif key == "hypoxic_fraction_map" and isinstance(value, dict):
                    report["reconstruction_hypoxic_fraction_map"] = self._json_safe(value)
                    for sub_key, sub_value in value.items():
                        report[f"reconstruction_hypoxic_{sub_key}"] = self._json_safe(sub_value)
                else:
                    report[f"reconstruction_{key}"] = self._json_safe(value)

        if "predicted_state" in report:
            report["predicted_state_label"] = self._english_state_label(report.get("predicted_state"))
        if "alert_level" in report:
            report["alert_label"] = self._english_alert_label(report.get("alert_level"))
        return report

    def build_publication_report_tex(self, gui) -> str:
        report = self.build_diagnostic_report(gui)
        state_label = self._english_state_label(report.get("predicted_state"))
        alert_label = self._english_alert_label(report.get("alert_level"))
        uncertainty_summary = report.get("reconstruction_uncertainty_summary", "not available")
        assumption_summary = report.get("reconstruction_assumption_summary", "No explicit Krogh-fit assumption summary is available for this case.")
        burden_summary = report.get("reconstruction_hypoxic_burden_summary", "No hidden hypoxic burden estimate is available for this case.")
        radius_summary = self._build_radius_summary(report)
        radius_condition_note = self._build_radius_condition_note(report)
        interpretation = self._build_interpretation_sections(report)
        follow_up = self._build_follow_up_recommendation(report)
        figure_path = str(report.get("reconstruction_report_figure_path", "") or "")
        normalized_figure_path = figure_path.replace(os.sep, "/")
        figure_section = ""
        if figure_path and os.path.exists(figure_path):
            figure_section = """
\\section*{Embedded Figure}
\\begin{center}
\\includegraphics[width=0.86\\linewidth]{%s}
\\end{center}
""" % normalized_figure_path

        return """\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\begin{document}

\\section*{Diagnostic Summary}
This report was generated automatically by the Krogh GUI. For documentation consistency, exported reports are always written in English.

\\section*{Input Data}
\\begin{tabular}{|l|l|}
\\hline
Parameter & Value \\\\ \\hline
Blood-gas PO2 & %s mmHg \\\\ \\hline
Blood-gas PCO2 & %s mmHg \\\\ \\hline
pH & %s \\\\ \\hline
Temperature & %s C \\\\ \\hline
Sensor PO2 & %s mmHg \\\\ \\hline
Hemoglobin & %s g/dL \\\\ \\hline
Venous saturation & %s %% \\\\ \\hline
\\end{tabular}

\\section*{Probabilistic Result}
Predicted state: %s\\
Alert level: %s\\
Risk score: %s\\
Confidence: %s\\
Certainty: %s\\
%s

\\section*{Clinical Interpretation}
%s

\\section*{Model-Based Findings}
%s

\\section*{Krogh reconstruction}
Fitted mitoP50: %s mmHg\\
Perfusion factor: %s x baseline\\
Simulated venous PO2: %s mmHg\\
Simulated sensor PO2: %s mmHg\\
Uncertainty band: %s

\\section*{Assumptions Required for Best Fit}
%s

\\section*{Estimated Hidden Hypoxic Burden}
%s

\\section*{Radius Sensitivity Across Assumed Tissue Radius}
%s

\\section*{Cautions and Limitations}
%s

\\section*{Suggested Follow-Up}
%s
%s
\\end{document}
""" % (
            self._latex_escape(report.get("po2", "")),
            self._latex_escape(report.get("pco2", "")),
            self._latex_escape(report.get("pH", "")),
            self._latex_escape(report.get("temperature_c", "")),
            self._latex_escape(report.get("sensor_po2", "")),
            self._latex_escape(report.get("hemoglobin_g_dl", "")),
            self._latex_escape(report.get("venous_sat_percent", "")),
            self._latex_escape(state_label),
            self._latex_escape(alert_label),
            self._latex_escape(report.get("risk_score", "n/a")),
            self._latex_escape(report.get("confidence", "n/a")),
            self._latex_escape(report.get("certainty", "n/a")),
            self._latex_escape(radius_condition_note),
            self._latex_escape(interpretation["clinical"]),
            self._latex_escape(interpretation["findings"]),
            self._latex_escape(report.get("reconstruction_P_half_fit", "n/a")),
            self._latex_escape(report.get("reconstruction_perfusion_factor", "n/a")),
            self._latex_escape(report.get("reconstruction_P_v_sim", "n/a")),
            self._latex_escape(report.get("reconstruction_sensor_sim", "n/a")),
            self._latex_escape(uncertainty_summary),
            self._latex_escape(assumption_summary),
            self._latex_escape(burden_summary),
            self._latex_escape(radius_summary),
            self._latex_escape(interpretation["cautions"]),
            self._latex_escape(follow_up),
            figure_section,
        )

    def build_publication_report_text(self, gui) -> str:
        report = self.build_diagnostic_report(gui)
        interpretation = self._build_interpretation_sections(report)
        follow_up = self._build_follow_up_recommendation(report)
        assumption_summary = report.get("reconstruction_assumption_summary", "No explicit Krogh-fit assumption summary is available for this case.")
        burden_summary = report.get("reconstruction_hypoxic_burden_summary", "No hidden hypoxic burden estimate is available for this case.")
        radius_summary = self._build_radius_summary(report)
        radius_condition_note = self._build_radius_condition_note(report)
        lines = [
            "Diagnostic Summary",
            "==================",
            "This exported report is intentionally fixed in English.",
            "",
            "Input Data",
            "----------",
            f"Blood-gas PO2: {report.get('po2', '')} mmHg",
            f"Blood-gas PCO2: {report.get('pco2', '')} mmHg",
            f"pH: {report.get('pH', '')}",
            f"Temperature: {report.get('temperature_c', '')} C",
            f"Sensor PO2: {report.get('sensor_po2', '')} mmHg",
            f"Hemoglobin: {report.get('hemoglobin_g_dl', '')} g/dL",
            f"Venous saturation: {report.get('venous_sat_percent', '')} %",
            "",
            "Probabilistic Result",
            "--------------------",
            f"Predicted state: {self._english_state_label(report.get('predicted_state'))}",
            f"Alert level: {self._english_alert_label(report.get('alert_level'))}",
            f"Risk score: {report.get('risk_score', 'n/a')}",
            f"Confidence: {report.get('confidence', 'n/a')}",
            f"Certainty: {report.get('certainty', 'n/a')}",
            radius_condition_note,
            "",
            "Clinical Interpretation",
            "-----------------------",
            interpretation["clinical"],
            "",
            "Model-Based Findings",
            "--------------------",
            interpretation["findings"],
            "",
            "Krogh reconstruction",
            "--------------------",
            f"Fitted mitoP50: {report.get('reconstruction_P_half_fit', 'n/a')} mmHg",
            f"Perfusion factor: {report.get('reconstruction_perfusion_factor', 'n/a')} x baseline",
            f"Simulated venous PO2: {report.get('reconstruction_P_v_sim', 'n/a')} mmHg",
            f"Simulated sensor PO2: {report.get('reconstruction_sensor_sim', 'n/a')} mmHg",
            f"Uncertainty band: {report.get('reconstruction_uncertainty_summary', 'not available')}",
            "",
            "Assumptions Required for Best Fit",
            "---------------------------------",
            str(assumption_summary),
            "",
            "Estimated Hidden Hypoxic Burden",
            "-------------------------------",
            str(burden_summary),
            "",
            "Radius Sensitivity Across Assumed Tissue Radius",
            "----------------------------------------------",
            str(radius_summary),
            "",
            "Cautions and Limitations",
            "------------------------",
            interpretation["cautions"],
            "",
            "Suggested Follow-Up",
            "-------------------",
            follow_up,
        ]
        figure_path = report.get('reconstruction_report_figure_path')
        if figure_path:
            lines.extend([
                "",
                "Embedded Figure",
                "---------------",
                f"Saved figure path: {figure_path}",
            ])
        return "\n".join(lines) + "\n"

    def save_publication_report(self, report_text: str, path: str) -> None:
        target_path = os.path.abspath(path)
        lower_path = target_path.lower()

        if lower_path.endswith(".pdf"):
            tex_path = os.path.splitext(target_path)[0] + ".tex"
            with open(tex_path, "w", encoding="utf-8") as handle:
                handle.write(report_text)

            compiler = shutil.which("pdflatex")
            if not compiler:
                raise RuntimeError("Automatic PDF generation requires pdflatex to be installed and available in PATH.")

            output_dir = os.path.dirname(tex_path) or os.getcwd()
            result = subprocess.run(
                [
                    compiler,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={output_dir}",
                    tex_path,
                ],
                capture_output=True,
                text=True,
            )

            generated_pdf = os.path.splitext(tex_path)[0] + ".pdf"
            if result.returncode != 0 or not os.path.exists(generated_pdf):
                details = (result.stderr or result.stdout or "Unknown LaTeX error").strip()
                raise RuntimeError(f"PDF generation failed: {details}")

            if os.path.abspath(generated_pdf) != target_path:
                shutil.move(generated_pdf, target_path)

            for suffix in (".aux", ".log", ".out"):
                aux_path = os.path.splitext(tex_path)[0] + suffix
                if os.path.exists(aux_path):
                    try:
                        os.remove(aux_path)
                    except OSError:
                        pass
            return

        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)

    def save_diagnostic_report(self, report: dict[str, Any], path: str) -> None:
        if str(path).lower().endswith(".csv"):
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(report.keys()))
                writer.writeheader()
                writer.writerow(report)
            return

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    def save_to_path(self, payload: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_from_path(self, path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def apply_case_payload(self, gui, data: dict[str, Any]) -> None:
        gui.mode_var.set("single")
        gui._toggle_inputs()

        for name, entry in gui.entries.items():
            if name in data:
                entry.config(state="normal")
                entry.delete(0, "end")
                entry.insert(0, str(data[name]))

        if "include_axial_diffusion" in data:
            gui.include_axial_var.set(bool(data["include_axial_diffusion"]))

        if "numeric_settings" in data and isinstance(data["numeric_settings"], dict):
            for key, entry in gui.numeric_entries.items():
                if key in data["numeric_settings"]:
                    entry.config(state="normal")
                    entry.delete(0, "end")
                    entry.insert(0, str(data["numeric_settings"][key]))

        if "diagnostic_settings" in data and isinstance(data["diagnostic_settings"], dict):
            for key, entry in gui.diagnostic_entries.items():
                if key in data["diagnostic_settings"]:
                    entry.config(state="normal")
                    entry.delete(0, "end")
                    entry.insert(0, str(data["diagnostic_settings"][key]))

        if hasattr(gui, "output") and "output_text" in data:
            gui.output.delete("1.0", "end")
            if data["output_text"]:
                gui.output.insert("1.0", str(data["output_text"]))

        if hasattr(gui, "diagnostic_output") and "diagnostic_output_text" in data:
            gui.diagnostic_output.config(state="normal")
            gui.diagnostic_output.delete("1.0", "end")
            if data["diagnostic_output_text"]:
                gui.diagnostic_output.insert("1.0", str(data["diagnostic_output_text"]))
            gui.diagnostic_output.config(state="disabled")

        if hasattr(gui, "diagnostic_radius_mode_var") and "diagnostic_radius_mode" in data:
            gui.diagnostic_radius_mode_var.set(str(data.get("diagnostic_radius_mode", "all variants")))
        if hasattr(gui, "diagnostic_radius_variant_var") and "diagnostic_radius_variant" in data:
            gui.diagnostic_radius_variant_var.set(str(data.get("diagnostic_radius_variant", "100 µm")))
        if hasattr(gui, "_toggle_diagnostic_radius_variant_controls"):
            gui._toggle_diagnostic_radius_variant_controls()

        gui.last_diagnostic_result = data.get("last_diagnostic_result")
        gui.last_krogh_reconstruction = data.get("last_krogh_reconstruction")

    def capture_ui_state(self, gui) -> dict[str, Any]:
        series_entry_keys = {
            "start_value": gui.t("start_value"),
            "end_value": gui.t("end_value"),
            "step_size": gui.t("step_size"),
            "secondary_start_value": gui.t("secondary_start_value"),
            "secondary_end_value": gui.t("secondary_end_value"),
            "secondary_step_size": gui.t("secondary_step_size"),
        }
        state = {
            "mode": gui.mode_var.get(),
            "include_axial": gui.include_axial_var.get(),
            "save_series_results": gui.save_series_results_var.get(),
            "lock_hypoxic_fraction_scale": gui.lock_hypoxic_fraction_scale_var.get(),
            "publication_mode": gui.publication_mode_var.get(),
            "publication_layout": gui.publication_layout_key,
            "series_dimension": gui.series_dimension_var.get(),
            "series_plot_mode": gui.series_plot_mode_var.get(),
            "series_param_key": gui.series_param_key,
            "series_param2_key": gui.series_param2_key,
            "entries": {},
            "series_entries": {},
            "numeric_entries": {},
            "output_text": "",
            "plot_selection": [],
            "diagnostic_entries": {},
            "diagnostic_output_text": "",
            "diagnostic_radius_mode": getattr(gui, "diagnostic_radius_mode_var", None).get() if hasattr(gui, "diagnostic_radius_mode_var") else "all variants",
            "diagnostic_radius_variant": getattr(gui, "diagnostic_radius_variant_var", None).get() if hasattr(gui, "diagnostic_radius_variant_var") else "100 µm",
        }
        if hasattr(gui, "entries"):
            state["entries"] = {name: entry.get() for name, entry in gui.entries.items()}
        if hasattr(gui, "series_entries"):
            state["series_entries"] = {
                key: gui.series_entries[label].get()
                for key, label in series_entry_keys.items()
                if label in gui.series_entries
            }
        if hasattr(gui, "numeric_entries"):
            state["numeric_entries"] = {key: entry.get() for key, entry in gui.numeric_entries.items()}
        if hasattr(gui, "output"):
            state["output_text"] = gui.output.get("1.0", "end-1c")
        if hasattr(gui, "series_plot_listbox"):
            state["plot_selection"] = list(gui.series_plot_listbox.curselection())
        if hasattr(gui, "diagnostic_entries"):
            state["diagnostic_entries"] = {key: entry.get() for key, entry in gui.diagnostic_entries.items()}
        if hasattr(gui, "diagnostic_output"):
            state["diagnostic_output_text"] = gui.diagnostic_output.get("1.0", "end-1c")
        if hasattr(gui, "series_param_display_to_key"):
            state["series_param_key"] = gui.series_param_display_to_key.get(gui.series_param_var.get(), gui.series_param_key)
            state["series_param2_key"] = gui.series_param_display_to_key.get(gui.series_param2_var.get(), gui.series_param2_key)
        return state

    def restore_ui_state(self, gui, state: dict[str, Any]) -> None:
        series_entry_keys = {
            "start_value": gui.t("start_value"),
            "end_value": gui.t("end_value"),
            "step_size": gui.t("step_size"),
            "secondary_start_value": gui.t("secondary_start_value"),
            "secondary_end_value": gui.t("secondary_end_value"),
            "secondary_step_size": gui.t("secondary_step_size"),
        }
        gui.mode_var.set(state["mode"])
        gui.include_axial_var.set(state["include_axial"])
        if hasattr(gui, "diagnostic_radius_mode_var"):
            gui.diagnostic_radius_mode_var.set(state.get("diagnostic_radius_mode", "all variants"))
        if hasattr(gui, "diagnostic_radius_variant_var"):
            gui.diagnostic_radius_variant_var.set(state.get("diagnostic_radius_variant", "100 µm"))
        gui.save_series_results_var.set(state.get("save_series_results", False))
        gui.lock_hypoxic_fraction_scale_var.set(state.get("lock_hypoxic_fraction_scale", True))
        gui.publication_mode_var.set(state.get("publication_mode", False))
        gui._set_publication_layout_display(state.get("publication_layout", "wide"))
        if hasattr(gui, "_toggle_diagnostic_radius_variant_controls"):
            gui._toggle_diagnostic_radius_variant_controls()
        gui.series_dimension_var.set(state.get("series_dimension", "1d"))
        gui.series_plot_mode_var.set(state.get("series_plot_mode", "2d"))
        gui.series_param_key = state["series_param_key"]
        gui.series_param2_key = state.get("series_param2_key", gui.series_param2_key)
        gui._set_series_param_display(gui.series_param_key)
        gui._set_series_param2_display(gui.series_param2_key)

        for name, value in state["entries"].items():
            if name in gui.entries:
                widget = gui.entries[name]
                if isinstance(widget, ttk.Combobox):
                    if name == "Relative_PO2_reference":
                        try:
                            internal_key = gui._parse_relative_reference_mode(value)
                            translated_label = gui.t(f"reference_{internal_key}")
                            widget.set(translated_label)
                        except ValueError:
                            widget.set(value)
                    else:
                        widget.set(value)
                else:
                    widget.delete(0, "end")
                    widget.insert(0, value)

        for key, value in state["series_entries"].items():
            label = series_entry_keys.get(key)
            if label in gui.series_entries:
                gui.series_entries[label].delete(0, "end")
                gui.series_entries[label].insert(0, value)

        for key, value in state.get("numeric_entries", {}).items():
            if key in gui.numeric_entries:
                gui.numeric_entries[key].delete(0, "end")
                gui.numeric_entries[key].insert(0, value)

        gui.series_plot_listbox.selection_clear(0, "end")
        for index in state["plot_selection"]:
            if 0 <= index < gui.series_plot_listbox.size():
                gui.series_plot_listbox.selection_set(index)

        if state["output_text"]:
            gui.output.insert("1.0", state["output_text"])

        for key, value in state.get("diagnostic_entries", {}).items():
            if key in getattr(gui, "diagnostic_entries", {}):
                gui.diagnostic_entries[key].delete(0, "end")
                gui.diagnostic_entries[key].insert(0, value)

        if state.get("diagnostic_output_text") and hasattr(gui, "diagnostic_output"):
            gui.diagnostic_output.config(state="normal")
            gui.diagnostic_output.delete("1.0", "end")
            gui.diagnostic_output.insert("1.0", state["diagnostic_output_text"])
            gui.diagnostic_output.config(state="disabled")

        gui._toggle_inputs()


__all__ = ["CaseRepository"]
