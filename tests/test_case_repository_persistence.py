import unittest

from krogh_app.persistence import CaseRepository


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyEntry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def delete(self, *args, **kwargs):
        self.value = ""

    def insert(self, index, value):
        self.value = str(value)

    def config(self, **kwargs):
        return None


class DummyText:
    def __init__(self, value=""):
        self.value = value
        self.state = "normal"

    def get(self, *args, **kwargs):
        return self.value

    def delete(self, *args, **kwargs):
        self.value = ""

    def insert(self, index, value):
        self.value = str(value)

    def config(self, **kwargs):
        self.state = kwargs.get("state", self.state)


class CaseRepositoryPersistenceTests(unittest.TestCase):
    def _make_gui(self):
        gui = type("DummyGUI", (), {})()
        gui.entries = {"PO2_inlet_mmHg": DummyEntry("80")}
        gui.numeric_entries = {"ode_rtol": DummyEntry("1e-6")}
        gui.diagnostic_entries = {"po2": DummyEntry("85")}
        gui.include_axial_var = DummyVar(True)
        gui.output = DummyText("general output")
        gui.diagnostic_output = DummyText("diagnostic output")
        gui.last_diagnostic_result = {"alert_level": "ok", "confidence": 0.9}
        gui.last_krogh_reconstruction = {
            "P_half_fit": 1.2,
            "perfusion_factor": 0.95,
            "uncertainty": {
                "p_half_low": 0.8,
                "p_half_high": 1.6,
                "perfusion_low": 0.7,
                "perfusion_high": 1.1,
            },
        }
        return gui

    def test_case_payload_includes_latest_outputs_and_reconstruction(self):
        repo = CaseRepository()
        gui = self._make_gui()

        payload = repo.build_case_payload(gui)

        self.assertEqual(payload["output_text"], "general output")
        self.assertEqual(payload["diagnostic_output_text"], "diagnostic output")
        self.assertEqual(payload["last_diagnostic_result"]["alert_level"], "ok")
        self.assertEqual(payload["last_krogh_reconstruction"]["uncertainty"]["p_half_low"], 0.8)


if __name__ == "__main__":
    unittest.main()
