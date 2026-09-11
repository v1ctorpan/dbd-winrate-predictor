import unittest

import torch

import match_model as mm


class TestMatchGRU(unittest.TestCase):
    def test_forward_shape(self):
        model = mm.MatchGRU()
        padded = torch.randn(2, 5, 30)
        lengths = torch.tensor([5, 3])
        logits = model.forward(padded, lengths)
        self.assertEqual(logits.shape, (2, 5))

    def test_predict_proba(self):
        model = mm.MatchGRU()
        padded = torch.randn(2, 5, 30)
        lengths = torch.tensor([5, 3])
        proba = model.predict_proba(padded, lengths)
        self.assertEqual(proba.shape, (2, 5))
        self.assertTrue(torch.allclose(proba.sum(dim=1), torch.ones(2), atol=1e-4))

    def test_single_step(self):
        model = mm.MatchGRU()
        feat = torch.randn(1, 30)
        logits, hidden = model.single_step(feat)
        self.assertEqual(logits.shape, (1, 5))
        self.assertEqual(hidden.shape, (1, 1, 64))
        logits2, _ = model.single_step(feat, hidden)
        self.assertEqual(logits2.shape, (1, 5))

    def test_variable_length_uses_full_sequence(self):
        model = mm.MatchGRU()
        padded = torch.randn(1, 8, 30)
        lengths = torch.tensor([8])
        full_logits = model.forward(padded, lengths)
        single_logits = model.forward(padded[:, :1], torch.tensor([1]))
        self.assertFalse(torch.allclose(full_logits, single_logits, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
