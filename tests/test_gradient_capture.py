import tempfile
import unittest
from pathlib import Path

import torch

from safety_project_a4.gradient_capture import apply_spf_projection, capture_reference_gradient


class GradientCaptureTest(unittest.TestCase):
    def test_chunked_projection_matches_dense_formula(self):
        torch.manual_seed(11)
        model = torch.nn.Linear(4, 7, bias=False)
        original = torch.randn_like(model.weight)
        basis, _ = torch.linalg.qr(torch.randn(7, 2), mode="reduced")
        model.weight.grad = original.clone()
        apply_spf_projection(model, {"weight": basis}, chunk_elements=8)
        expected = original - basis @ (basis.T @ original)
        torch.testing.assert_close(model.weight.grad, expected, rtol=1e-6, atol=1e-6)

    def test_capture_preserves_utility_gradient_and_projects(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2, bias=False)
        utility_loss = model(torch.tensor([[1.0, 2.0, -1.0]])).square().mean()
        utility_loss.backward()
        before = model.weight.grad.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            result = capture_reference_gradient(
                model,
                lambda: model(torch.tensor([[0.5, -1.0, 2.0]])).sum(),
                compute_dot=True,
                compute_bases=True,
                spool_root=Path(directory),
                spool_label="test",
                rank=1,
                oversampling=0,
                niter=1,
                allowlist_policy="all_2d",
            )
            torch.testing.assert_close(model.weight.grad, before)
            self.assertIsNotNone(result.spool)
            spooled = result.spool.read("weight", device=model.weight.device)
            self.assertEqual(spooled.shape, model.weight.shape)
            apply_spf_projection(model, result.bases)
            basis = result.bases["weight"]
            residual = basis.T @ model.weight.grad.float()
            torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=1e-5)
            result.spool.cleanup()


if __name__ == "__main__":
    unittest.main()
