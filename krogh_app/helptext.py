"""Text builders for GUI help content."""

from __future__ import annotations

from typing import Any, Callable


class HelpTextBuilder:
    """Builds explanatory texts for output and numeric settings help windows."""

    def build_output_parameter_help_text(
        self,
        *,
        language_code: str,
        t: Callable[..., str],
        result_label: Callable[[str], str],
    ) -> str:
        if language_code == "de":
            lines = [
                t("output_help_intro"),
                "",
                "Aktueller Modellstand:",
                "- Nichtnegative PO2-Werte werden auf 0 mmHg begrenzt.",
                "- Der Gewebeverbrauch folgt einer Michaelis-Menten-Kinetik mit basal verbleibendem Restverbrauch von 5 % des Maximalverbrauchs.",
                "- Die axiale Gewebediffusion kann zugeschaltet werden und koppelt Gewebefeld und Kapillarprofil iterativ.",
                "",
                f"{result_label('P50_eff')}: effektiver Hb-P50 nach pH-, pCO2- und Temperatur-Shift. Ein hoeherer Wert bedeutet geringere Hb-Affinität.",
                f"{result_label('P_venous')}: kapillärer PO2 am venösen Ende des axialen Profils.",
                f"{result_label('P_tissue_min')}: globales Minimum des gesamten Gewebefelds. Sehr sensitiv, sättigt bei schwerer Hypoxie aber rasch bei 0.",
                f"{result_label('P_tissue_p05')}: 5. Perzentil aller Gewebe-PO2-Werte. Robuster als das absolute Minimum und gut für kritische Hypoxie geeignet.",
                f"{result_label('Hypoxic_fraction_lt1')}: Anteil des Gewebegitters mit PO2 < 1 mmHg. Marker extremer anoxischer Areale.",
                f"{result_label('Hypoxic_fraction_lt5')}: Anteil des Gewebegitters mit PO2 < 5 mmHg. Marker ausgeprägter kritischer Hypoxie.",
                f"{result_label('Hypoxic_fraction_lt10')}: Anteil des Gewebegitters mit PO2 < 10 mmHg. Klinisch weichere Frühwarnschwelle für Versorgungseinbußen.",
                f"{result_label('PO2_fraction_gt100')}: Anteil des Gewebegitters mit PO2 ueber dem ersten frei waehlbaren Schwellenwert. Marker hyperoxischer Gewebebereiche.",
                f"{result_label('PO2_fraction_gt200')}: Anteil des Gewebegitters mit PO2 ueber dem zweiten frei waehlbaren Schwellenwert. Marker ausgepraegter Hyperoxie.",
                f"{result_label('PO2_fraction_gt_rel1')}: Anteil des Gewebegitters ueber dem ersten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{result_label('PO2_fraction_gt_rel2')}: Anteil des Gewebegitters ueber dem zweiten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{result_label('PO2_fraction_gt_rel3')}: Anteil des Gewebegitters ueber dem dritten relativen Schwellenwert bezogen auf PO2_inlet.",
                f"{result_label('PO2_sensor_avg')}: radial volumen-gewichteter und axial gemittelter Gewebe-PO2. Entspricht am ehesten einem gemittelten Sensorsignal.",
                f"{result_label('S_a_percent')}: arterielle Hb-Sättigung aus Einlass-PO2 und P50_eff.",
                f"{result_label('S_v_percent')}: venöse Hb-Sättigung aus Endkapillar-PO2 und P50_eff.",
                f"{result_label('Q_flow_nL_s')}: lokaler Kapillarfluss nach Skalierung mit dem Perfusionsfaktor.",
            ]
        else:
            lines = [
                t("output_help_intro"),
                "",
                "Current model status:",
                "- Non-negative PO2 values are clipped at 0 mmHg.",
                "- Tissue consumption follows Michaelis-Menten kinetics with a residual basal demand of 5 % of maximal consumption.",
                "- Optional axial tissue diffusion couples the tissue field and capillary profile iteratively.",
                "",
                f"{result_label('P50_eff')}: effective hemoglobin P50 after pH, pCO2, and temperature shift.",
                f"{result_label('P_venous')}: capillary PO2 at the venous end of the axial profile.",
                f"{result_label('P_tissue_min')}: global minimum of the full tissue field. Very sensitive, but it saturates at 0 in severe hypoxia.",
                f"{result_label('P_tissue_p05')}: 5th percentile of all tissue PO2 values. More robust than the absolute minimum.",
                f"{result_label('Hypoxic_fraction_lt1')}: fraction of the tissue grid with PO2 < 1 mmHg.",
                f"{result_label('Hypoxic_fraction_lt5')}: fraction of the tissue grid with PO2 < 5 mmHg.",
                f"{result_label('Hypoxic_fraction_lt10')}: fraction of the tissue grid with PO2 < 10 mmHg.",
                f"{result_label('PO2_fraction_gt100')}: fraction of the tissue grid above the first user-defined threshold.",
                f"{result_label('PO2_fraction_gt200')}: fraction of the tissue grid above the second user-defined threshold.",
                f"{result_label('PO2_fraction_gt_rel1')}: fraction of the tissue grid above the first inlet-relative threshold.",
                f"{result_label('PO2_fraction_gt_rel2')}: fraction of the tissue grid above the second inlet-relative threshold.",
                f"{result_label('PO2_fraction_gt_rel3')}: fraction of the tissue grid above the third inlet-relative threshold.",
                f"{result_label('PO2_sensor_avg')}: radially volume-weighted and axially averaged tissue PO2, closest to a mean sensor signal.",
                f"{result_label('S_a_percent')}: arterial Hb saturation from inlet PO2 and P50_eff.",
                f"{result_label('S_v_percent')}: venous Hb saturation from end-capillary PO2 and P50_eff.",
                f"{result_label('Q_flow_nL_s')}: local capillary flow after scaling with the perfusion factor.",
            ]
        return "\n".join(lines)

    def build_numeric_parameter_help_text(
        self,
        *,
        language_code: str,
        t: Callable[..., str],
        numeric_settings_fields: tuple,
        get_numeric_spec: Callable[[str], dict[str, Any]],
        numeric_label: Callable[[str], str],
        format_numeric_value: Callable[[str, Any], str],
        current_value_getter: Callable[[str], str],
    ) -> str:
        lines = [t("numeric_help_intro"), ""]
        for key, label_key, _ in numeric_settings_fields:
            spec = get_numeric_spec(key)
            lines.extend(
                [
                    numeric_label(label_key),
                    spec.get("description_de", "") if language_code == "de" else spec.get("description_en", ""),
                    t(
                        "numeric_help_range_line",
                        min_value=format_numeric_value(key, spec["min"]),
                        max_value=format_numeric_value(key, spec["max"]),
                        default_value=format_numeric_value(key, spec["default"]),
                    ),
                    t(
                        "numeric_help_current_line",
                        current_value=current_value_getter(key) or format_numeric_value(key, spec["default"]),
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip()


__all__ = ["HelpTextBuilder"]
