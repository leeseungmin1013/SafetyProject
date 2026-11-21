import unittest

import torch

from safety_project_a4.optimizer import InspectableAdamW


class OptimizerTest(unittest.TestCase):
    def test_matches_torch_adamw_for_two_steps(self):
        ours = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32))
        reference = torch.nn.Parameter(ours.detach().clone())
        custom = InspectableAdamW(
            [("weight", ours)], lr=2e-3, betas=(0.5, 0.999), eps=1e-8, weight_decay=0.01
        )
        torch_optimizer = torch.optim.AdamW(
            [reference],
            lr=2e-3,
            betas=(0.5, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            foreach=False,
            fused=False,
        )
        for gradient in (
            torch.tensor([0.2, -0.1, 0.4]),
            torch.tensor([-0.3, 0.6, 0.2]),
        ):
            ours.grad = gradient.clone()
            reference.grad = gradient.clone()
            custom.step()
            torch_optimizer.step()
            torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-7)

    def test_probe_reports_realized_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
        optimizer = InspectableAdamW([("weight", parameter)], lr=1e-3, state_dtype="fp32")
        parameter.grad = torch.tensor([0.2, -0.1], dtype=torch.bfloat16)
        captured = {}
        optimizer.step(probe_callback=lambda name, grad, stages: captured.update(stages))
        self.assertIn("d_actual", captured)
        self.assertIn("d_m_reset", captured)
        self.assertIn("d_v_reset", captured)
        self.assertIn("d_no_spf_adamw", captured)
        self.assertEqual(captured["d_actual"].dtype, torch.float32)

    def test_no_spf_counterfactual_does_not_mutate_real_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
        optimizer = InspectableAdamW(
            [("weight", parameter)], lr=1e-2, betas=(0.5, 0.999), weight_decay=0.0
        )
        parameter.grad = torch.tensor([0.25, -0.5])
        captured = {}
        optimizer.step(
            probe_callback=lambda name, grad, stages: captured.update(stages),
            no_spf_gradient_provider=lambda name, device: torch.tensor(
                [-0.75, 0.1], device=device
            ),
        )
        self.assertFalse(torch.allclose(captured["d_no_spf_adamw"], captured["d_adamw"]))
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], torch.tensor([0.125, -0.25]))


if __name__ == "__main__":
    unittest.main()
