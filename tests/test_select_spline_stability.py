"""Minimal checks of the declared validation-only continuation rule."""
import copy
import unittest
from scripts.select_spline_stability import select_candidate


class SelectionTests(unittest.TestCase):
    def fixture(self):
        reference = dict(provenance=dict(sample_ids=['fixed'], evaluation_protocol={}, input='image', dataset_root='same', weights_unchanged=True),
                         common=dict(samples=1952, common_curve_f1=.84, common_branch_f1=.27),
                         native=dict(node_f1=.91, edge_f1=.86),
                         matched=dict(all_pairs=dict(endpoint_neutral_guide_curve_mm=.36)))
        return reference, dict(curve_f1_margin=.01, node_f1_margin=.01, shape_error_ratio=1.05)

    def test_geometry_guard_precedes_branch_ranking(self):
        reference, settings = self.fixture()
        candidates = {a:copy.deepcopy(reference) for a in ['control','good','damaged']}
        candidates['good']['common']['common_branch_f1'] = .30
        candidates['damaged']['common'].update(common_branch_f1=.40, common_curve_f1=.70)
        selected, decisions = select_candidate(candidates,reference,settings)
        self.assertEqual(selected,'good')
        self.assertFalse(decisions['damaged']['eligible'])

    def test_no_eligible_candidate_stops_extension(self):
        reference, settings = self.fixture()
        damaged = copy.deepcopy(reference)
        damaged['matched']['all_pairs']['endpoint_neutral_guide_curve_mm'] = .50
        with self.assertRaises(RuntimeError):
            select_candidate({'damaged':damaged},reference,settings)


if __name__=='__main__':unittest.main()
