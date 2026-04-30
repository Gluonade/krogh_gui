import unittest

from krogh_app.benchmarking import (
    ReconstructionBenchmarkCase,
    ReconstructionBenchmarkRunner,
    build_default_reconstruction_benchmark_cases,
)


class _FakeReconstructor:
    def fit_joint_parameters(self, **kwargs):
        strategy = kwargs.get("search_strategy", "optimized")
        objective = 1.25 if strategy == "optimized" else 1.40
        candidate_count = 5 if strategy == "optimized" else 14
        return {
            "objective": objective,
            "perfusion_factor": 0.92,
            "P_half_fit": 1.10,
            "metabolic_rate_rel": 1.05,
            "sensor_error": 0.8,
            "venous_error": 1.2,
            "fit_warning": False,
            "fit_boundary_hit": False,
            "uncertainty": {
                "candidate_count": candidate_count,
                "identifiability": "moderate",
            },
        }


class ReconstructionBenchmarkingTests(unittest.TestCase):
    def test_default_benchmark_cases_cover_normoxia_edge_and_stress(self):
        cases = build_default_reconstruction_benchmark_cases()

        self.assertGreaterEqual(len(cases), 7)
        categories = {case.category for case in cases}
        self.assertIn("normoxia", categories)
        self.assertIn("edge", categories)
        self.assertIn("stress", categories)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_runner_emits_comparison_rows(self):
        runner = ReconstructionBenchmarkRunner(
            cases=[
                ReconstructionBenchmarkCase(
                    case_id="reference",
                    label="Reference",
                    category="normoxia",
                    P_inlet=80.0,
                    sensor_target=25.0,
                    P_v_target=32.0,
                    pH=7.4,
                    pCO2=40.0,
                    temp_c=37.0,
                )
            ],
            reconstructor=_FakeReconstructor(),
        )

        results = runner.run(verbose=False)

        self.assertIn("cases", results)
        self.assertIn("legacy_cases", results)
        self.assertIn("comparison", results)
        self.assertIn("comparison_summary", results)
        self.assertEqual(len(results["cases"]), 1)
        self.assertEqual(len(results["legacy_cases"]), 1)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertEqual(results["cases"][0]["search_strategy"], "optimized")
        self.assertEqual(results["legacy_cases"][0]["search_strategy"], "legacy_grid")
        self.assertEqual(results["cases"][0]["category"], "normoxia")
        self.assertEqual(results["comparison"][0]["category"], "normoxia")
        self.assertGreater(results["comparison_summary"]["mean_speedup_ratio"], 0.0)
        self.assertIn("speedup_ratio", results["comparison"][0])


if __name__ == "__main__":
    unittest.main()