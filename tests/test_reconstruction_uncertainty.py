import unittest

import numpy as np

from krogh_app.reconstruction import KroghReconstructor


class ReconstructionUncertaintyTests(unittest.TestCase):
    def _build_reconstructor(self):
        radial_weights = np.array([1.0, 1.0, 1.0])
        r_vec = np.array([0.0, 0.5, 1.0])
        z_eval = np.array([0.0, 0.5, 1.0])

        def fake_solver(P_inlet, P_half, p50_eff, include_axial_diffusion, perfusion_factor):
            venous = P_inlet - 4.5 * np.log1p(P_half) - 2.0 * abs(perfusion_factor - 1.0)
            tissue_level = max(0.1, P_inlet - 8.0 * np.log1p(P_half) - 3.0 / max(perfusion_factor, 0.1))
            P_c_axial = np.array([P_inlet, 0.5 * (P_inlet + venous), venous], dtype=float)
            tissue_po2 = np.full((len(z_eval), len(r_vec)), tissue_level, dtype=float)
            return P_c_axial, tissue_po2, None

        return KroghReconstructor(
            solve_axial_capillary_po2=fake_solver,
            effective_p50=lambda **kwargs: 26.0,
            radial_weights=radial_weights,
            r_vec=r_vec,
            z_eval=z_eval,
            R_cap=1.0,
            R_tis=2.0,
            L_cap=3.0,
        )

    def test_joint_fit_returns_uncertainty_summary(self):
        reconstructor = self._build_reconstructor()

        fit = reconstructor.fit_joint_parameters(
            P_inlet=90.0,
            sensor_target=70.0,
            P_v_target=55.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
        )

        self.assertIn("uncertainty", fit)
        uncertainty = fit["uncertainty"]
        self.assertGreaterEqual(uncertainty["candidate_count"], 1)
        self.assertLessEqual(uncertainty["p_half_low"], fit["P_half_fit"])
        self.assertGreaterEqual(uncertainty["p_half_high"], fit["P_half_fit"])
        self.assertLessEqual(uncertainty["perfusion_low"], fit["perfusion_factor"])
        self.assertGreaterEqual(uncertainty["perfusion_high"], fit["perfusion_factor"])

    def test_uncertainty_reports_bootstrap_interval_metadata(self):
        reconstructor = self._build_reconstructor()

        fit = reconstructor.fit_joint_parameters(
            P_inlet=90.0,
            sensor_target=70.0,
            P_v_target=55.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
            bootstrap_samples=24,
        )

        uncertainty = fit["uncertainty"]
        self.assertIn("method", uncertainty)
        self.assertIn("bootstrap_samples", uncertainty)
        self.assertIn("bootstrap_successes", uncertainty)
        self.assertIn("p_half_p10", uncertainty)
        self.assertIn("p_half_p90", uncertainty)
        self.assertIn("perfusion_p10", uncertainty)
        self.assertIn("perfusion_p90", uncertainty)
        self.assertEqual(uncertainty["bootstrap_samples"], 24)
        self.assertGreaterEqual(uncertainty["bootstrap_successes"], 10)
        self.assertLessEqual(uncertainty["p_half_p10"], uncertainty["p_half_p90"])
        self.assertLessEqual(uncertainty["perfusion_p10"], uncertainty["perfusion_p90"])
        self.assertIn("bootstrap", uncertainty["summary"].lower())

    def test_uncertainty_reports_identifiability_metadata(self):
        reconstructor = self._build_reconstructor()

        fit = reconstructor.fit_joint_parameters(
            P_inlet=90.0,
            sensor_target=70.0,
            P_v_target=55.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
            bootstrap_samples=16,
        )

        uncertainty = fit["uncertainty"]
        self.assertIn("identifiability", uncertainty)
        self.assertIn("identifiability_summary", uncertainty)
        self.assertIn("parameter_correlation", uncertainty)
        self.assertIn("profile_p_half_low", uncertainty)
        self.assertIn("profile_p_half_high", uncertainty)
        self.assertIn("profile_perfusion_low", uncertainty)
        self.assertIn("profile_perfusion_high", uncertainty)
        self.assertIn("sensitivity_matrix", uncertainty)
        self.assertIn("fisher_determinant", uncertainty)
        self.assertIn("fisher_condition", uncertainty)
        self.assertIn(uncertainty["identifiability"], {"strong", "moderate", "weak"})
        self.assertLessEqual(uncertainty["profile_p_half_low"], uncertainty["profile_p_half_high"])
        self.assertLessEqual(uncertainty["profile_perfusion_low"], uncertainty["profile_perfusion_high"])
        self.assertGreaterEqual(uncertainty["fisher_determinant"], 0.0)
        self.assertIn("identifiability", uncertainty["identifiability_summary"].lower())

    def test_plot_data_contains_hidden_hypoxic_burden_summary(self):
        reconstructor = self._build_reconstructor()
        fit = reconstructor.fit_joint_parameters(
            P_inlet=30.0,
            sensor_target=8.0,
            P_v_target=12.0,
            pH=7.25,
            pCO2=55.0,
            temp_c=37.0,
        )

        plot_data = reconstructor.build_plot_data(
            po2=30.0,
            pco2=55.0,
            ph=7.25,
            temperature_c=37.0,
            venous_sat=55.0,
            P_v_target=12.0,
            venous_weight=0.5,
            sensor_po2=8.0,
            diag_result={"predicted_state": "severe_hypoxia", "alert_level": "red"},
            fit=fit,
        )

        self.assertIn("hypoxic_burden_summary", plot_data)
        self.assertIn("assumption_summary", plot_data)
        self.assertIn("below_10", plot_data["hypoxic_fraction_map"])
        self.assertGreaterEqual(plot_data["hypoxic_fraction_map"]["below_10"], 0.0)
        self.assertLessEqual(plot_data["hypoxic_fraction_map"]["below_10"], 1.0)

    def test_plot_data_contains_radius_sensitivity_scenarios(self):
        reconstructor = self._build_reconstructor()
        fit = reconstructor.fit_joint_parameters(
            P_inlet=45.0,
            sensor_target=12.0,
            P_v_target=20.0,
            pH=7.3,
            pCO2=48.0,
            temp_c=37.0,
        )

        plot_data = reconstructor.build_plot_data(
            po2=45.0,
            pco2=48.0,
            ph=7.3,
            temperature_c=37.0,
            venous_sat=60.0,
            P_v_target=20.0,
            venous_weight=0.4,
            sensor_po2=12.0,
            diag_result={"predicted_state": "compensated_hypoxia", "alert_level": "yellow"},
            fit=fit,
        )

        self.assertIn("radius_scenarios", plot_data)
        self.assertIn("radius_sensitivity_summary", plot_data)
        self.assertIn("normal_30um", plot_data["radius_scenarios"])
        self.assertIn("high_100um", plot_data["radius_scenarios"])
        self.assertIn("30 µm", plot_data["radius_sensitivity_summary"])
        self.assertIn("alert_level", plot_data["radius_scenarios"]["normal_30um"])
        self.assertIn("alert_level", plot_data["radius_scenarios"]["high_100um"])

    def test_joint_fit_enforces_sensor_mean_target(self):
        reconstructor = self._build_reconstructor()

        fit = reconstructor.fit_joint_parameters(
            P_inlet=90.0,
            sensor_target=70.0,
            P_v_target=55.0,
            pH=7.4,
            pCO2=40.0,
            temp_c=37.0,
            venous_weight=1.5,
        )

        self.assertAlmostEqual(float(fit["sensor_sim"]), 70.0, places=2)
        self.assertLessEqual(float(fit["sensor_error"]), 0.02)


if __name__ == "__main__":
    unittest.main()
