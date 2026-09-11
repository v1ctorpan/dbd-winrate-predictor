import json
import os
import tempfile
import unittest

import torch

import train_sequence as ts


def _frame(t=0):
    return [0, 0, 0, 0, 0, 0, 0, 0, 5, t]


def _write_videos(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _rec(id_, match, label, n=4):
    return {"id": id_, "title": "", "url": "", "match": match,
            "frames": [_frame(t=i * 2) for i in range(n)], "label": label}


class TestTrain(unittest.TestCase):
    def test_split_by_match(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [_rec("BV%d" % i, 1, i) for i in range(5)])
            from match_dataset import MatchDataset
            ds = MatchDataset(p)
            train_idxs, val_idxs = ts.split_matches(ds, val_frac=0.2, seed=42)
            self.assertEqual(len(val_idxs), 1)
            self.assertEqual(len(train_idxs), 4)
            self.assertTrue(set(train_idxs).isdisjoint(set(val_idxs)))

    def test_train_loop_runs(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [_rec("BV1", 1, 0), _rec("BV16", 1, 1)])
            from match_dataset import MatchDataset
            ds = MatchDataset(p)
            train_idxs, _ = ts.split_matches(ds, val_frac=0.0, seed=42)
            model, opt, criterion = ts.make_training(torch.device("cpu"))
            loss = ts.train_one_epoch(model, ds, train_idxs, opt, criterion,
                                      batch_size=2, device=torch.device("cpu"))
            self.assertGreater(loss, 0.0)

    def test_save_load_ckpt(self):
        model = ts.make_model()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            torch.save(model.state_dict(), ckpt)
            model2 = ts.make_model()
            model2.load_state_dict(torch.load(ckpt, weights_only=True))
            self.assertTrue(os.path.exists(ckpt))

    def test_train_always_saves_checkpoint(self):
        """即使 val_acc 为 0，训练结束也必须留下 checkpoint（否则无法推理）。"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [_rec("BV1", 1, 0), _rec("BV16", 1, 1)])
            ckpt = os.path.join(d, "m.pt")
            ts.train(p, ckpt, epochs=1, seed=0)
            self.assertTrue(os.path.exists(ckpt), "checkpoint not saved")


if __name__ == "__main__":
    unittest.main()
