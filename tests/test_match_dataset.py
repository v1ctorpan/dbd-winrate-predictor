import json
import os
import tempfile
import unittest

import torch

import match_dataset as md


def _frame(state_idx=0, gens=5, t=0, hooks=0):
    return [state_idx, state_idx, state_idx, state_idx,
            hooks, hooks, hooks, hooks, gens, t]


def _write_videos(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(id_, match, label, frames):
    return {"id": id_, "title": "", "url": "", "match": match,
            "frames": frames, "label": label}


class TestMatchDataset(unittest.TestCase):
    def test_filters_unlabeled_and_len(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [
                _rec("BV1", 1, 3, [_frame(t=0), _frame(t=20)]),
                _rec("BV1", 2, -1, [_frame(t=0)]),
                _rec("BV2", 1, 0, [_frame(t=0)]),
            ])
            ds = md.MatchDataset(p)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.keys, ["BV1:1", "BV2:1"])

    def test_getitem_shape_and_time_zero_based(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [
                _rec("BV1", 1, 3, [_frame(state_idx=1, t=40), _frame(state_idx=2, t=60)]),
            ])
            ds = md.MatchDataset(p)
            feats, label = ds[0]
            self.assertIsInstance(feats, torch.Tensor)
            self.assertEqual(feats.shape, (2, 30))
            self.assertEqual(label, 3)
            self.assertAlmostEqual(float(feats[0, 29]), 0.0, places=6)
            self.assertAlmostEqual(float(feats[1, 29]), 20 / 1200.0, places=6)

    def test_collate_padding(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            _write_videos(p, [
                _rec("BV1", 1, 3, [_frame(t=i * 2) for i in range(5)]),
                _rec("BV2", 1, 0, [_frame(t=i * 2) for i in range(3)]),
            ])
            ds = md.MatchDataset(p)
            padded, lengths, labels = md.collate_fn([ds[0], ds[1]])
            self.assertEqual(padded.shape, (2, 5, 30))
            self.assertEqual(lengths.tolist(), [5, 3])
            self.assertEqual(labels.tolist(), [3, 0])

    def test_truncated_item_from_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            frames = [_frame(gens=g, t=i * 2) for i, g in enumerate([5, 5, 4, 3, 2])]
            _write_videos(p, [_rec("BV1", 1, 3, frames)])
            ds = md.MatchDataset(p)
            feats, label = md.truncated_item(ds, 0, k=2)
            self.assertEqual(feats.shape, (3, 30))
            self.assertEqual(label, 3)
            # 第 2 帧 gens=4 -> one-hot 后 gens 位置在 index 28
            self.assertAlmostEqual(float(feats[0, 28]), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
