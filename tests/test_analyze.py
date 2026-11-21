import unittest

from safety_project_a4.analyze import analyze_records


def metric(classification, dot):
    return {
        "dot": dot,
        "reference_norm": 1.0,
        "vector_norm": 1.0,
        "cosine": dot,
        "tau": 1e-6,
        "classification": classification,
    }


class AnalyzeTest(unittest.TestCase):
    def test_aggregates_conflicts_events_and_delta_ce(self):
        stages = {
            "d_raw": metric("neutral", 0.0),
            "d_m": metric("conflict", 0.2),
            "d_pre": metric("conflict", 0.3),
            "d_adamw": metric("conflict", 0.3),
            "d_actual": metric("conflict", 0.3),
            "d_m_reset": metric("neutral", 0.0),
            "d_v_reset": metric("conflict", 0.2),
            "d_no_spf_adamw": metric("conflict", 0.5),
        }
        events = {
            "spf_residual": False,
            "momentum_induced": True,
            "preconditioner_induced": False,
            "decay_induced": False,
            "realized_update_residual": False,
        }
        records = [
            {
                "run_id": "example-run",
                "step": 1,
                "spf_triggered": True,
                "fresh": stages,
                "audit": stages,
                "fresh_events": events,
                "audit_events": events,
                "audit_panel_delta_ce": 0.1,
            },
            {
                "run_id": "example-run",
                "step": 2,
                "spf_triggered": False,
                "fresh": stages,
                "audit": stages,
                "fresh_events": events,
                "audit_events": events,
                "audit_panel_delta_ce": -0.05,
            },
        ]
        summary = analyze_records(records)
        self.assertEqual(summary["probe_count"], 2)
        self.assertEqual(summary["spf_trigger_rate"], 0.5)
        self.assertEqual(
            summary["references"]["fresh"]["events"]["momentum_induced"]["rate"], 1.0
        )
        self.assertEqual(summary["audit_panel_delta_ce"]["positive_n"], 1)


if __name__ == "__main__":
    unittest.main()

