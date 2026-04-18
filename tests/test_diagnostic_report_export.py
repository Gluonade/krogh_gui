import csv
import json
import os
import tempfile
import unittest

from krogh_app.persistence import CaseRepository


class DummyEntry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class DiagnosticReportExportTests(unittest.TestCase):
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
            "uncertainty": {
                "p_half_low": 0.9,
                "p_half_high": 1.8,
                "perfusion_low": 0.8,
                "perfusion_high": 1.4,
            },
        }
        return gui

    def test_build_diagnostic_report_contains_reconstruction_uncertainty(self):
        repo = CaseRepository()
        report = repo.build_diagnostic_report(self._make_gui())

        self.assertEqual(report["predicted_state"], "normoxia")
        self.assertEqual(report["alert_level"], "ok")
        self.assertEqual(report["reconstruction_P_half_fit"], 1.3)
        self.assertEqual(report["reconstruction_uncertainty_p_half_low"], 0.9)

    def test_save_diagnostic_report_writes_json_and_csv(self):
        repo = CaseRepository()
        report = repo.build_diagnostic_report(self._make_gui())

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "diag_report.json")
            csv_path = os.path.join(tmpdir, "diag_report.csv")
            repo.save_diagnostic_report(report, json_path)
            repo.save_diagnostic_report(report, csv_path)

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["predicted_state"], "normoxia")

            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alert_level"], "ok")


if __name__ == "__main__":
    unittest.main()
