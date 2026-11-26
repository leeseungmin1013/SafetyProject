import unittest

from a2_runner import agreement, probe_rows


class A2Test(unittest.TestCase):
    def test_probes_do_not_reuse_rows(self):
        probes = probe_rows(100, 3, 4, 17)
        self.assertEqual(len(probes), 3)
        self.assertEqual(len({i for probe in probes for i in probe}), 12)
        self.assertEqual(probes, probe_rows(100, 3, 4, 17))

    def test_agreement_includes_class_breakdown(self):
        self.assertEqual(agreement([True, False], [False, False]), {
            'correct': 1, 'total': 2, 'rate': .5,
            'batch_conflict_rate': .5, 'conflict': 0., 'no_conflict': 1.,
        })
