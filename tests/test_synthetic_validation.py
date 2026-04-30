import csv
import os
import tempfile
import unittest

from krogh_app.validation import SyntheticValidationRunner, build_default_synthetic_cases, build_extended_synthetic_cases


class SyntheticValidationTests(unittest.TestCase):
    def test_default_suite_has_representative_categories(self):
        cases = build_default_synthetic_cases()
        extended_cases = build_extended_synthetic_cases()

        self.assertGreaterEqual(len(cases), 6)
        self.assertGreaterEqual(len(extended_cases), 9)
        categories = {case.category for case in cases}
        extended_categories = {case.category for case in extended_cases}
        self.assertIn("normoxia", categories)
        self.assertIn("mild_hypoxia", categories)
        self.assertIn("severe_hypoxia", categories)
        self.assertIn("low_perfusion", categories)
        self.assertIn("anemia_stress", extended_categories)
        self.assertIn("temperature_stress", extended_categories)

    def test_runner_returns_summary_and_case_rows(self):
        runner = SyntheticValidationRunner(cases=build_default_synthetic_cases()[:2])
        results = runner.run_default_suite(sensor_noise_std=0.0, venous_sat_noise_std=0.0, bootstrap_samples=2)

        self.assertIn("summary", results)
        self.assertIn("cases", results)
        self.assertEqual(results["summary"]["case_count"], len(results["cases"]))
        self.assertGreaterEqual(results["summary"]["pass_count"], 2)
        self.assertGreaterEqual(results["summary"]["reconstruction_success_rate"], 0.90)
        self.assertLessEqual(results["summary"]["mean_sensor_error"], 3.0)
        self.assertIn("case_id", results["cases"][0])
        self.assertIn("status", results["cases"][0])
        self.assertIn("status_reason", results["cases"][0])
        self.assertIn("predicted_state", results["cases"][0])
        self.assertTrue(results["cases"][0]["status_reason"])

    def test_runner_can_save_csv_report(self):
        runner = SyntheticValidationRunner(cases=build_default_synthetic_cases()[:2])
        results = runner.run_default_suite(sensor_noise_std=0.0, venous_sat_noise_std=0.0, bootstrap_samples=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "synthetic_validation_report.csv")
            runner.save_csv_report(results, csv_path)

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), results["summary"]["case_count"])
            self.assertIn("case_id", rows[0])
            self.assertIn("status", rows[0])

    def test_trend_checks_follow_expected_direction(self):
        runner = SyntheticValidationRunner(cases=build_default_synthetic_cases()[:1])
        trend = runner.run_trend_checks()

        self.assertIn("summary", trend)
        self.assertIn("checks", trend)
        self.assertEqual(trend["summary"]["check_count"], 2)
        self.assertTrue(trend["summary"]["perfusion_sensor_monotonic"])
        self.assertTrue(trend["summary"]["inlet_sensor_monotonic"])


if __name__ == "__main__":
    unittest.main()
