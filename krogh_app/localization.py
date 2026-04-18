"""Localization helpers for the Krogh GUI project."""

from __future__ import annotations

from typing import Mapping


class TranslationManager:
    """Resolves translated UI text and localized field labels."""

    _STATE_LABELS = {
        "en": {
            "normoxia": "normoxia",
            "mild_hypoxia": "mild tissue hypoxia",
            "compensated_hypoxia": "compensated tissue hypoxia",
            "severe_hypoxia": "severe tissue hypoxia",
            "profound_hypoxia": "profound tissue hypoxia",
        },
        "de": {
            "normoxia": "Normoxie",
            "mild_hypoxia": "milde Gewebehypoxie",
            "compensated_hypoxia": "kompensierte Gewebehypoxie",
            "severe_hypoxia": "schwere Gewebehypoxie",
            "profound_hypoxia": "ausgepraegte Gewebehypoxie",
        },
    }

    _RESULT_DESCRIPTIONS = {
        "de": {
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
        },
        "en": {
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
        },
    }

    def __init__(
        self,
        *,
        translations: Mapping[str, Mapping[str, str]],
        input_field_labels: Mapping[str, str],
        result_field_labels: Mapping[str, str],
    ):
        self.translations = {key: dict(value) for key, value in translations.items()}
        self.input_field_labels = dict(input_field_labels)
        self.result_field_labels = dict(result_field_labels)

    def _normalize_language_code(self, language_code: str | None) -> str:
        token = str(language_code or "en").strip().lower()
        aliases = {
            "english": "en",
            "deutsch": "de",
            "german": "de",
            "francais": "fr",
            "français": "fr",
            "italiano": "it",
            "espanol": "es",
            "español": "es",
        }
        return aliases.get(token, token if token in self.translations else "en")

    def translate(self, key: str, *, language_code: str | None = None, **kwargs) -> str:
        code = self._normalize_language_code(language_code)
        language_map = self.translations.get(code, self.translations.get("en", {}))
        template = language_map.get(key, self.translations.get("en", {}).get(key, key))
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def field_label(self, field_name: str, *, language_code: str | None = None) -> str:
        label_key = self.input_field_labels.get(field_name, field_name)
        return self.translate(label_key, language_code=language_code)

    def numeric_label(self, field_name: str, *, language_code: str | None = None) -> str:
        return self.translate(field_name, language_code=language_code)

    def bool_label(self, value: bool, *, language_code: str | None = None) -> str:
        return self.translate("bool_yes" if value else "bool_no", language_code=language_code)

    def parse_relative_reference_mode(self, raw_value: str, *, language_code: str | None = None) -> str:
        token = str(raw_value).strip().lower().replace(" ", "").replace("_", "")
        if token in {"", "inlet"}:
            return "inlet"
        if token in {"tissue_max", "tissuemax", "tissue", "max_tissue", "maxtissue"}:
            return "tissue_max"

        code = self._normalize_language_code(language_code)
        for key in ["inlet", "tissue_max"]:
            translated = self.translate(f"reference_{key}", language_code=code).strip().lower().replace(" ", "").replace("_", "")
            if translated == token:
                return key

        if token in {"gewebemax", "maxgewebe"}:
            return "tissue_max"
        raise ValueError

    def result_label(
        self,
        field_name: str,
        *,
        language_code: str | None = None,
        absolute_thresholds: tuple[float, float] = (100.0, 200.0),
        relative_thresholds: tuple[float, float, float] = (90.0, 50.0, 30.0),
        relative_reference: str = "inlet",
    ) -> str:
        if field_name == "PO2_fraction_gt100":
            return self.translate(
                "result_po2_fraction_gt_primary",
                language_code=language_code,
                threshold=f"{float(absolute_thresholds[0]):.6g}",
            )
        if field_name == "PO2_fraction_gt200":
            return self.translate(
                "result_po2_fraction_gt_secondary",
                language_code=language_code,
                threshold=f"{float(absolute_thresholds[1]):.6g}",
            )
        if field_name == "PO2_fraction_gt_rel1":
            return self.translate(
                "result_po2_fraction_gt_rel_primary",
                language_code=language_code,
                threshold=f"{float(relative_thresholds[0]):.6g}",
                reference=self.translate(f"reference_{relative_reference}", language_code=language_code),
            )
        if field_name == "PO2_fraction_gt_rel2":
            return self.translate(
                "result_po2_fraction_gt_rel_secondary",
                language_code=language_code,
                threshold=f"{float(relative_thresholds[1]):.6g}",
                reference=self.translate(f"reference_{relative_reference}", language_code=language_code),
            )
        if field_name == "PO2_fraction_gt_rel3":
            return self.translate(
                "result_po2_fraction_gt_rel_tertiary",
                language_code=language_code,
                threshold=f"{float(relative_thresholds[2]):.6g}",
                reference=self.translate(f"reference_{relative_reference}", language_code=language_code),
            )
        label_key = self.result_field_labels.get(field_name, field_name)
        return self.translate(label_key, language_code=language_code)

    def result_description(self, field_name: str, *, language_code: str | None = None) -> str:
        code = self._normalize_language_code(language_code)
        descriptions = self._RESULT_DESCRIPTIONS.get("de" if code == "de" else "en", self._RESULT_DESCRIPTIONS["en"])
        return descriptions.get(field_name, self.result_label(field_name, language_code=code))

    def format_oxygenation_state_label(self, state_name: str, *, language_code: str | None = None) -> str:
        code = self._normalize_language_code(language_code)
        labels = self._STATE_LABELS.get("de" if code == "de" else "en", self._STATE_LABELS["en"])
        return labels.get(state_name, str(state_name).replace("_", " "))


__all__ = ["TranslationManager"]
