import unittest
from unittest.mock import patch

import krogh_GUI


class ReconstructionBenchmarkGuiTests(unittest.TestCase):
    def test_compute_reconstruction_benchmark_appends_summary_and_resets_progress(self):
        gui = krogh_GUI.KroghGUI.__new__(krogh_GUI.KroghGUI)
        appended_messages = []
        progress_states = []

        translations = {
            "diag_benchmark_ready": "Reconstruction benchmark finished.",
            "diag_benchmark_error": "ERROR in reconstruction benchmark: {error}",
        }

        gui.reconstructor = object()
        gui._append_async = appended_messages.append
        gui._set_progress_running = progress_states.append
        gui._call_on_ui_thread = lambda callback, *args, **kwargs: callback(*args, **kwargs)
        gui.t = lambda key, **kwargs: translations[key].format(**kwargs)

        fake_results = {
            "summary": {"case_count": 2, "mean_elapsed_s": 12.5, "median_elapsed_s": 12.0, "max_elapsed_s": 13.0, "warning_count": 0, "boundary_hit_count": 0},
            "comparison_summary": {"case_count": 2, "mean_speedup_ratio": 2.4, "mean_elapsed_saved_s": 8.1},
        }

        with patch.object(krogh_GUI, "run_and_save_default_reconstruction_benchmark", return_value=fake_results), patch.object(
            krogh_GUI.ReconstructionBenchmarkRunner,
            "format_summary_text",
            return_value="Reconstruction benchmark summary: 2 cases | mean 12.500 s | median 12.000 s | max 13.000 s | warnings 0 | boundary hits 0 | mean legacy speedup 2.40x.",
        ):
            gui._compute_reconstruction_benchmark()

        self.assertIn("Reconstruction benchmark finished.", appended_messages)
        self.assertTrue(any(message.startswith("[Benchmark] Reconstruction benchmark summary:") for message in appended_messages))
        self.assertTrue(any("Legacy comparison: mean speedup 2.40x" in message for message in appended_messages))
        self.assertTrue(any(message.startswith("[Benchmark] Saved reports to:") for message in appended_messages))
        self.assertEqual(progress_states[-1], False)


if __name__ == "__main__":
    unittest.main()