import os
import tempfile
import unittest

from krogh_app.persistence import CaseRepository


class DummyEntry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class PublicationReportExportTests(unittest.TestCase):
    def _make_gui(self):
        gui = type("DummyGUI", (), {})()
        gui.diagnostic_entries = {
            "po2": DummyEntry("82"),
            "pco2": DummyEntry("41"),
            "pH": DummyEntry("7.39"),
            "temperature_c": DummyEntry("37.0"),
            "sensor_po2": DummyEntry("64"),
            "hemoglobin_g_dl": DummyEntry("13.5"),
            "venous_sat_percent": DummyEntry("74"),
            "yellow_threshold": DummyEntry("0.4"),
            "orange_threshold": DummyEntry("0.6"),
            "red_threshold": DummyEntry("0.8"),
        }
        gui.last_diagnostic_result = {
            "predicted_state": "normoxia",
            "alert_level": "ok",
            "risk_score": 0.18,
            "confidence": 0.92,
            "certainty": 0.88,
        }
        gui.last_krogh_reconstruction = {
            "P_half_fit": 1.3,
            "perfusion_factor": 1.1,
            "P_v_sim": 53.2,
            "sensor_sim": 63.7,
            "uncertainty": {
                "p_half_low": 0.9,
                "p_half_high": 1.8,
                "perfusion_low": 0.8,
                "perfusion_high": 1.4,
                "summary": "mitoP50 ≈ 0.9–1.8 mmHg | perfusion ≈ 0.8–1.4x",
                "identifiability_summary": "identifiability appears moderate with local Fisher condition 14 and limited parameter coupling.",
            },
        }
        return gui

    def test_publication_report_is_english_and_contains_summary(self):
        repo = CaseRepository()
        tex = repo.build_publication_report_tex(self._make_gui())

        self.assertIn("Diagnostic Summary", tex)
        self.assertIn("Predicted state", tex)
        self.assertIn("Krogh reconstruction", tex)
        self.assertIn("Uncertainty band", tex)
        self.assertIn("normoxia", tex)

    def test_publication_report_contains_structured_interpretation_sections(self):
        repo = CaseRepository()
        gui = self._make_gui()
        gui.last_diagnostic_result = {
            "predicted_state": "low_oxygenation_approaching_critical",
            "alert_level": "orange",
            "risk_score": 0.67,
            "confidence": 0.79,
            "certainty": 0.31,
        }

        text = repo.build_publication_report_text(gui)

        self.assertIn("Clinical Interpretation", text)
        self.assertIn("Model-Based Findings", text)
        self.assertIn("Cautions and Limitations", text)
        self.assertIn("Moderate concern", text)
        self.assertIn("low tissue oxygen approaching critical values", text)

    def test_publication_report_contains_follow_up_recommendation(self):
        repo = CaseRepository()
        gui = self._make_gui()
        gui.last_diagnostic_result = {
            "predicted_state": "low_oxygenation_approaching_critical",
            "alert_level": "orange",
            "risk_score": 0.67,
            "confidence": 0.55,
            "certainty": 0.21,
        }

        text = repo.build_publication_report_text(gui)

        self.assertIn("Suggested Follow-Up", text)
        self.assertIn("repeat measurements", text.lower())
        self.assertIn("short-interval reassessment", text.lower())

    def test_publication_report_contains_identifiability_summary(self):
        repo = CaseRepository()
        text = repo.build_publication_report_text(self._make_gui())

        self.assertIn("identifiability", text.lower())
        self.assertIn("local fisher condition", text.lower())

    def test_radius_condition_note_reflects_actual_highest_alert(self):
        repo = CaseRepository()
        report = {
            "reconstruction_radius_scenarios": {
                "normal_30um": {
                    "radius_um": 30.0,
                    "alert_level": "red",
                    "fraction_below_1": 0.02,
                    "fraction_below_5": 0.20,
                    "fraction_below_10": 0.40,
                    "fraction_below_15": 0.50,
                },
                "increased_50um": {
                    "radius_um": 50.0,
                    "alert_level": "red",
                    "fraction_below_1": 0.02,
                    "fraction_below_5": 0.15,
                    "fraction_below_10": 0.30,
                    "fraction_below_15": 0.40,
                },
                "high_100um": {
                    "radius_um": 100.0,
                    "alert_level": "orange",
                    "fraction_below_1": 0.00,
                    "fraction_below_5": 0.05,
                    "fraction_below_10": 0.22,
                    "fraction_below_15": 0.30,
                },
            }
        }

        note = repo._build_radius_condition_note(report)
        self.assertIn("highest alert appears for 30", note)
        self.assertIn("50", note)
        self.assertNotIn("mainly appears under the larger/swollen-tissue", note)

    def test_publication_report_contains_assumption_and_burden_sections(self):
        repo = CaseRepository()
        gui = self._make_gui()
        gui.last_krogh_reconstruction = {
            "P_half_fit": 1.8,
            "perfusion_factor": 0.72,
            "P_v_sim": 21.4,
            "sensor_sim": 18.2,
            "hypoxic_fraction_map": {
                "below_1": 0.02,
                "below_5": 0.11,
                "below_10": 0.28,
                "below_15": 0.41,
            },
            "hypoxic_burden_summary": "Estimated hidden hypoxic burden: 28.0% of the tissue cylinder is below 10 mmHg and 11.0% is below 5 mmHg.",
            "assumption_summary": "To reproduce the measured pattern, the best fit required perfusion near 0.72 x baseline.",
            "radius_sensitivity_summary": "Radius sensitivity across assumed tissue radius: 30 µm gives a clearly milder burden than 100 µm.",
            "radius_scenarios": {
                "normal_30um": {"radius_um": 30.0, "fraction_below_10": 0.02},
                "increased_50um": {"radius_um": 50.0, "fraction_below_10": 0.11},
                "high_100um": {"radius_um": 100.0, "fraction_below_10": 0.28},
            },
            "uncertainty": {
                "summary": "mitoP50 ≈ 1.0–2.4 mmHg | perfusion ≈ 0.6–0.9x",
            },
        }

        text = repo.build_publication_report_text(gui)

        self.assertIn("Assumptions Required for Best Fit", text)
        self.assertIn("Estimated Hidden Hypoxic Burden", text)
        self.assertIn("Radius Sensitivity Across Assumed Tissue Radius", text)
        self.assertIn("30 µm", text)
        self.assertIn("100 µm", text)
        self.assertIn("green", text.lower())
        self.assertIn("yellow", text.lower())
        self.assertIn("below 10 mmHg", text)
        self.assertIn("best fit required", text)
        self.assertIn("conditional on the assumed tissue radius", text.lower())
        self.assertIn("larger/swollen-tissue", text.lower())
        self.assertIn("clinically plausible", text.lower())

    def test_publication_report_can_be_saved_to_tex(self):
        repo = CaseRepository()
        tex = repo.build_publication_report_tex(self._make_gui())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "publication_report.tex")
            repo.save_publication_report(tex, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("\\section*{Diagnostic Summary}", content)


if __name__ == "__main__":
    unittest.main()
