import unittest

import numpy as np
import torch

from scpp import select_representatives
from scpp.scoring import nearest_scores
from research.scpp.cp_select import prune_subset
from research.geometry.cross_core import synthetic_control as cross_control
from research.oracle.learning_core import synthetic_control as oracle_control


class SCPPControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_escape_rule_keeps_less_replaceable_endpoint(self):
        # Closest pair: 0 and .125. The .125 point is closer to its
        # alternative at10, so it is removed; the endpoint at0 survives.
        x = np.array([[0.], [.125], [10.], [30.]], dtype=np.float16)
        ids, _ = prune_subset(x,np.arange(4),3,'cpu',4)
        self.assertEqual(ids.tolist(),[0,2,3])

    def test_escape_tie_removes_larger_global_id(self):
        x = np.array([[0.], [0.], [10.], [30.]], dtype=np.float16)
        ids, _ = prune_subset(x,np.array([99,5,7,8]),3,'cpu',4)
        self.assertEqual(ids.tolist(),[5,7,8])

    def test_seeded_subset_adapter_matches_original_schedule(self):
        features = np.random.RandomState(9).normal(size=(80,8)).astype(np.float16)
        actual, _ = select_representatives(features,target=20,subsets=4,query_chunk=13)
        rng = np.random.RandomState(42)
        expected = []
        for _ in range(4):
            eligible = np.setdiff1d(np.arange(80),expected)
            sampled = rng.choice(eligible,size=20,replace=False)
            kept, _ = prune_subset(features[sampled],sampled,5,'cpu',13)
            expected.extend(kept.tolist())
        np.testing.assert_array_equal(actual,expected)
        replay, _ = select_representatives(features,target=20,subsets=4,query_chunk=13)
        np.testing.assert_array_equal(actual,replay)
        self.assertEqual(len(np.unique(actual)),20)

    def test_pool_contract_rejects_invalid_input(self):
        for a in [np.zeros((8,2),np.float32),np.full((8,2),np.nan,np.float16)]:
            with self.assertRaises(ValueError):
                select_representatives(a,target=4,subsets=2)

    def test_native_score_against_direct_small_integer_distances(self):
        q = np.array([[0,0],[3,4],[-2,5],[8,-1],[1,1]],np.float32)
        b = np.array([[0,0],[2,2],[7,-2]],np.float16)
        direct = np.sqrt(((q.astype(np.float64)[:,None]-b.astype(np.float64)[None])**2).sum(2).min(1)/2)
        score = nearest_scores(q,b,query_chunk=2)
        # A predeclared toy arithmetic bound, not a historical-map tolerance.
        np.testing.assert_allclose(score,direct,rtol=0,atol=1e-6)

    def test_original_oracle_sparse_gradient_and_unique_projection(self):
        self.assertEqual(oracle_control()['status'],'pass')

    def test_original_bidirectional_geometry_against_direct_distances(self):
        self.assertEqual(cross_control()['status'],'pass')


if __name__ == '__main__':
    unittest.main()
