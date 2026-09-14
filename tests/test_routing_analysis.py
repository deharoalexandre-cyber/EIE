"""Small independent receipts to test conservation and calibration/test separation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_routing import validate, phase_curves


def receipt():
    prefill = {'callbacks': 3, 'first_accesses': 2, 'reuse_distance': [0, 1, 0, 0],
               'experts': [{'expert': 0, 'selected': 2, 'hits': 1, 'misses': 1},
                           {'expert': 1, 'selected': 1, 'hits': 0, 'misses': 1}]}
    decode = {'callbacks': 3, 'first_accesses': 1, 'reuse_distance': [0, 1, 1, 0],
              'experts': [{'expert': 0, 'selected': 1, 'hits': 1, 'misses': 0},
                          {'expert': 1, 'selected': 1, 'hits': 0, 'misses': 1},
                          {'expert': 2, 'selected': 1, 'hits': 0, 'misses': 1}]}
    return {'prompt_tokens': 3, 'predict': 4, 'ews': {'hits': 2, 'misses': 4, 'callbacks': 6},
            'routing': {'schema': 'eie.routing/v1', 'enabled': True, 'outcome': 'complete',
                        'expert_count': 4, 'top_k': 1, 'slots': 2,
                        'layers': [{'layer': 1, 'expert_weight_bytes': 64, 'prefill': prefill, 'decode': decode}]}}


class AnalysisTests(unittest.TestCase):
    def test_valid_and_distances(self):
        hist = validate(receipt())
        c = phase_curves(hist, 'decode')
        self.assertEqual(c['serial_lru_hit_rate'], [0, 0, 1/3, 2/3, 2/3])
        self.assertEqual(c['selection_coverage'], [0, 1/3, 2/3, 1, 1])
        self.assertEqual(c['expert_weight_bytes'], [0, 64, 128, 192, 256])

    def test_calibration_does_not_see_test(self):
        hist = validate(receipt())
        calibration = copy.deepcopy(hist)
        calibration['layers'][0]['decode']['experts'] = [{'expert': 3, 'selected': 100}]
        c = phase_curves(hist, 'decode', calibration)
        self.assertEqual(c['selection_coverage'][1], 0)
        self.assertEqual(c['ranking']['1'][0], 3)

    def test_reject_corruption(self):
        for change in ['total', 'phase', 'distance', 'negative', 'id', 'partial', 'duplicate']:
            with self.subTest(change=change):
                r = receipt()
                p = r['routing']['layers'][0]['decode']
                if change == 'total': r['ews']['hits'] += 1
                if change == 'phase': p['callbacks'] += 1
                if change == 'distance': p['reuse_distance'][0] += 1
                if change == 'negative': p['first_accesses'] = -1
                if change == 'id': p['experts'][0]['expert'] = 4
                if change == 'partial': r['routing']['outcome'] = 'cancelled'
                if change == 'duplicate': r['routing']['layers'].append(copy.deepcopy(r['routing']['layers'][0]))
                with self.assertRaises(ValueError): validate(r)

    def test_layout_mismatch(self):
        h = validate(receipt())
        other = copy.deepcopy(h)
        other['layers'][0]['expert_weight_bytes'] += 1
        with self.assertRaises(ValueError): phase_curves(h, 'decode', other)


if __name__ == '__main__':
    unittest.main()
