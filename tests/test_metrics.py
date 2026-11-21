import unittest

from safety_project_a4.metrics import direction_metric, transition_events


class MetricsTest(unittest.TestCase):
    def test_norm_scaled_classification(self):
        self.assertEqual(direction_metric(2e-6, 1.0, 1.0).classification, "conflict")
        self.assertEqual(direction_metric(-2e-6, 1.0, 1.0).classification, "safe")
        self.assertEqual(direction_metric(5e-7, 1.0, 1.0).classification, "neutral")

    def test_transition_events(self):
        metrics = {
            "d_raw": direction_metric(0.0, 1.0, 1.0),
            "d_m": direction_metric(2e-6, 1.0, 1.0),
            "d_pre": direction_metric(3e-6, 1.0, 1.0),
            "d_adamw": direction_metric(3e-6, 1.0, 1.0),
            "d_actual": direction_metric(3e-6, 1.0, 1.0),
        }
        events = transition_events(metrics)
        self.assertTrue(events["momentum_induced"])
        self.assertFalse(events["preconditioner_induced"])


if __name__ == "__main__":
    unittest.main()

