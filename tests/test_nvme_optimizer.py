import tempfile
import unittest
from pathlib import Path

import torch

from safety_project_a4.gradient_capture import GradientSpool
from safety_project_a4.nvme_optimizer import NVMeInspectableAdamW


class NVMeOptimizerTests(unittest.TestCase):
    def test_two_steps_match_torch_adamw_across_chunks(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                ours = torch.nn.Parameter(torch.tensor([1., -2., .5, 3., -.7], dtype=dtype))
                reference = torch.nn.Parameter(ours.detach().clone())
                optimizer = NVMeInspectableAdamW(
                    [("weight", ours)], state_root=directory, run_id="parity", chunk_elements=2,
                    lr=2e-3, betas=(0.5, 0.999), eps=1e-8, weight_decay=0.01,
                )
                standard = torch.optim.AdamW(
                    [reference], lr=2e-3, betas=(0.5, 0.999), eps=1e-8,
                    weight_decay=0.01, foreach=False, fused=False,
                )
                for values in ([.2, -.1, .4, .3, -.2], [-.3, .6, .2, -.4, .7]):
                    gradient = torch.tensor(values, dtype=dtype)
                    ours.grad = gradient.clone()
                    reference.grad = gradient.clone()
                    optimizer.step()
                    standard.step()
                    tolerance = 1e-6 if dtype == torch.float32 else 2e-2
                    torch.testing.assert_close(ours, reference, rtol=tolerance, atol=tolerance)
                state = optimizer.state[ours]
                self.assertEqual(state["step"], 2)
                self.assertNotIn("exp_avg", state)
                self.assertEqual(sum(p.stat().st_size for p in optimizer.state_dir.iterdir()), ours.numel() * 8)
                optimizer.cleanup()
                self.assertFalse(optimizer.state_dir.exists())

    def test_probe_counterfactual_and_actual_delta_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            parameter = torch.nn.Parameter(torch.tensor([1., -1., .5, 2., -2.]))
            optimizer = NVMeInspectableAdamW(
                [("weight", parameter)], state_root=directory, run_id="probe", chunk_elements=2,
                lr=1e-2, weight_decay=0.0,
            )
            parameter.grad = torch.tensor([.25, -.5, .75, -.1, .2])
            before = parameter.detach().clone()
            captured = {key: [] for key in (
                "d_no_spf_adamw", "d_raw", "d_m", "d_pre", "d_adamw",
                "d_m_reset", "d_v_reset", "d_actual",
            )}
            def callback(name, start, end, stages):
                self.assertEqual(name, "weight")
                for stage, vector in stages:
                    self.assertEqual(vector.numel(), end - start)
                    captured[stage].append(vector.clone())
            optimizer.step(
                probe_callback=callback,
                no_spf_gradient_provider=lambda name, start, end, device: -parameter.grad[start:end],
            )
            for stage in captured:
                self.assertEqual(sum(chunk.numel() for chunk in captured[stage]), parameter.numel())
            actual = torch.cat(captured["d_actual"])
            torch.testing.assert_close(actual, parameter.detach() - before)
            self.assertFalse(torch.allclose(
                torch.cat(captured["d_no_spf_adamw"]), torch.cat(captured["d_adamw"])
            ))
            stored = torch.frombuffer(
                bytearray(next(optimizer.state_dir.iterdir()).read_bytes()), dtype=torch.float32
            )
            torch.testing.assert_close(stored[:5], .5 * parameter.grad)
            torch.testing.assert_close(stored[5:], .001 * parameter.grad.square())
            optimizer.cleanup()

    def test_bf16_probe_reports_realized_rounded_update(self):
        with tempfile.TemporaryDirectory() as directory:
            parameter = torch.nn.Parameter(torch.tensor([1., 2., -.5], dtype=torch.bfloat16))
            optimizer = NVMeInspectableAdamW(
                [("weight", parameter)], state_root=directory, run_id="bf16-probe",
                chunk_elements=2, lr=1e-2,
            )
            parameter.grad = torch.tensor([.2, -.1, .4], dtype=torch.bfloat16)
            before = parameter.detach().float().clone()
            captured = []
            def callback(name, start, end, stages):
                for stage, vector in stages:
                    if stage == "d_actual":
                        captured.append(vector.clone())
            optimizer.step(probe_callback=callback)
            torch.testing.assert_close(torch.cat(captured), parameter.detach().float() - before)
            optimizer.cleanup()

    def test_probe_and_normal_steps_keep_identical_real_adamw_state(self):
        with tempfile.TemporaryDirectory() as directory:
            probed = torch.nn.Parameter(torch.tensor([1., -2., .5, 3., -.7]))
            normal = torch.nn.Parameter(probed.detach().clone())
            probe_optimizer = NVMeInspectableAdamW(
                [("weight", probed)], state_root=directory, run_id="with-probes",
                chunk_elements=2, lr=2e-3, betas=(0.5, 0.999), weight_decay=0.01,
            )
            normal_optimizer = NVMeInspectableAdamW(
                [("weight", normal)], state_root=directory, run_id="without-probes",
                chunk_elements=2, lr=2e-3, betas=(0.5, 0.999), weight_decay=0.01,
            )
            for values in ([.2, -.1, .4, .3, -.2], [-.3, .6, .2, -.4, .7]):
                probed.grad = torch.tensor(values)
                normal.grad = probed.grad.clone()
                probe_optimizer.step(probe_callback=lambda name, start, end, stages: list(stages))
                normal_optimizer.step()
                torch.testing.assert_close(probed, normal, rtol=1e-6, atol=1e-7)
            probe_state = next(probe_optimizer.state_dir.iterdir()).read_bytes()
            normal_state = next(normal_optimizer.state_dir.iterdir()).read_bytes()
            self.assertEqual(probe_state, normal_state)
            probe_optimizer.cleanup()
            normal_optimizer.cleanup()

    def test_gradient_spool_reads_only_requested_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = GradientSpool(Path(directory), "spool", chunk_elements=2)
            values = torch.tensor([1., -.5, 3., 2., -4.], dtype=torch.bfloat16)
            spool.write("weight", values)
            torch.testing.assert_close(spool.read_slice("weight", 1, 4, device=torch.device("cpu")), values[1:4])
            torch.testing.assert_close(spool.read("weight", device=torch.device("cpu")), values)
            self.assertEqual(next(spool.path.iterdir()).stat().st_size, values.numel() * values.element_size())
            spool.cleanup()


if __name__ == "__main__":
    unittest.main()
