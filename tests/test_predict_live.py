import os
import tempfile
import unittest

import torch

import predict_live as pl


def _make_ckpt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from match_model import MatchGRU
    torch.save(MatchGRU().state_dict(), path)


class TestLivePredictor(unittest.TestCase):
    def _row(self, frame="frame_01_00.0.jpg"):
        return {"frame": frame, "p1": "healthy", "p2": "injured",
                "p3": "hooked", "p4": "dying", "hooks": "1/2/0/1", "gens": "3"}

    def test_update_returns_proba(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            p = pred.update(self._row())
            self.assertEqual(len(p), 5)
            self.assertAlmostEqual(sum(p), 1.0, places=4)

    def test_stateful_accumulation(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            r1 = pred.update(self._row("frame_00_00.0.jpg"))
            r2 = pred.update(self._row("frame_01_00.0.jpg"))
            self.assertEqual(len(r1), 5)
            self.assertEqual(len(r2), 5)
            self.assertEqual(pred.n, 2)

    def test_reset(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            _make_ckpt(ckpt)
            pred = pl.LivePredictor(ckpt)
            pred.update(self._row())
            pred.reset()
            self.assertIsNone(pred.hidden)
            self.assertIsNone(pred.t0)
            p = pred.update(self._row())
            self.assertEqual(len(p), 5)

    def test_row_features_time_zero_based(self):
        row = self._row("frame_02_00.0.jpg")
        feats = pl.row_features(row, t0_halfsec=240)  # 120s = 240 half-sec
        self.assertEqual(len(feats), 30)
        self.assertAlmostEqual(feats[29], 0.0, places=6)
        row2 = self._row("frame_03_00.0.jpg")
        feats2 = pl.row_features(row2, t0_halfsec=240)
        self.assertAlmostEqual(feats2[29], 120 / 1200.0, places=6)


if __name__ == "__main__":
    unittest.main()
