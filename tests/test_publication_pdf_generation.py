import os
import tempfile
import unittest

from krogh_app.persistence import CaseRepository


class DummyEntry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class PublicationPdfGenerationTests(unittest.TestCase):
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
                "summary": "mitoP50 ≈ 0.9–1.8 mmHg | perfusion ≈ 0.8–1.4x",
            },
        }
        return gui

    def test_save_publication_report_generates_pdf(self):
        repo = CaseRepository()
        tex = repo.build_publication_report_tex(self._make_gui())

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "publication_report.pdf")
            repo.save_publication_report(tex, pdf_path)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 0)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "publication_report.tex")))


if __name__ == "__main__":
    unittest.main()
