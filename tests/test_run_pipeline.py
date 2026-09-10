import json
import os
import tempfile
import unittest
from unittest import mock

import run_pipeline as rp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")


class TestPipeline(unittest.TestCase):
    def test_frames_dir_pipeline_writes_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            videos = os.path.join(d, "videos.jsonl")
            report_root = os.path.join(d, "report")
            frames_root = os.path.join(d, "frames")
            meta = {"title": "测试标题", "url": "https://www.bilibili.com/video/BV1Uu8z6eEVM"}
            stats = rp.run_frames_dir(SRC, "BV1Uu8z6eEVM", sample=16,
                                      videos=videos, report_root=report_root,
                                      frames_root=frames_root, meta=meta)
            self.assertGreaterEqual(stats["matches"], 1)
            with open(videos, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertGreaterEqual(len(lines), 1)
            self.assertEqual(lines[0]["id"], "BV1Uu8z6eEVM")
            self.assertEqual(lines[0]["title"], "测试标题")
            self.assertEqual(lines[0]["url"], "https://www.bilibili.com/video/BV1Uu8z6eEVM")
            self.assertEqual(len(lines[0]["features"][0]), 30)

    def test_video_stats_report_closed_matches_not_dataset_total(self):
        class FakeDet:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def feed(self, frame, fname):
                self.calls += 1
                return {"match_end": 1} if self.calls == 1 else None

            def finish(self):
                return [2]

        frames = [(None, "frame_00_00.0.jpg"), (None, "frame_00_00.5.jpg")]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rp, "_iter_video_frames", return_value=iter(frames)), \
                mock.patch.object(rp.sd, "StreamingDetector", FakeDet), \
                mock.patch.object(rp, "_encode_match", return_value=1):
            videos = os.path.join(d, "videos.jsonl")
            open(videos, "w").close()
            stats = rp.run_video("dummy.mp4", "BV1X", videos=videos,
                                 report_root=d, frames_root=d)
        self.assertEqual(stats["matches"], 2)
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["closed"], [1, 2])

    def test_manual_single_match_uses_fixed_anchor_and_end_time(self):
        class FakeDet:
            instance = None

            def __init__(self, *args, **kwargs):
                self.anchor_prior = kwargs["anchor_prior"]
                self.detect_match_end = kwargs["detect_match_end"]
                FakeDet.instance = self

            def feed(self, frame, fname):
                return None

            def finish(self):
                return [1]

        anchor = {"x": 141, "y": 804, "scale": 1.6}
        frames = [(None, "frame_00_00.0.jpg"), (None, "frame_00_00.5.jpg")]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rp, "_iter_window_frames", return_value=iter(frames)) as it, \
                mock.patch.object(rp.sd, "StreamingDetector", FakeDet), \
                mock.patch.object(rp, "_encode_match", return_value=1):
            stats = rp.run_video_manual("dummy.mp4", "BV1X", anchor, 850.5,
                                        videos=os.path.join(d, "videos.jsonl"),
                                        report_root=d, frames_root=d)
        it.assert_called_once_with("dummy.mp4", 0.0, 850.5, 0.5)
        self.assertEqual(FakeDet.instance.anchor_prior, anchor)
        self.assertFalse(FakeDet.instance.detect_match_end)
        self.assertEqual(stats, {"matches": 1, "records": 1, "closed": [1]})

    def test_manual_single_match_auto_detects_anchor(self):
        class FakeDet:
            instance = None

            def __init__(self, *args, **kwargs):
                self.anchor_prior = kwargs["anchor_prior"]
                FakeDet.instance = self

            def feed(self, frame, fname):
                return None

            def finish(self):
                return [1]

        class FakePre:
            anchor = {"x": 141, "y": 804, "scale": 1.6}

        frames = [(None, "frame_00_00.0.jpg")]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rp, "_video_duration", return_value=100.0), \
                mock.patch.object(rp.prescan, "run_prescan", return_value=FakePre()) as rpre, \
                mock.patch.object(rp, "_iter_window_frames", return_value=iter(frames)) as it, \
                mock.patch.object(rp.sd, "StreamingDetector", FakeDet), \
                mock.patch.object(rp, "_encode_match", return_value=1):
            stats = rp.run_video_manual("dummy.mp4", "BV1X",
                                        videos=os.path.join(d, "videos.jsonl"),
                                        report_root=d, frames_root=d)
        rpre.assert_called_once()
        it.assert_called_once_with("dummy.mp4", 0.0, 100.0, 0.5)
        self.assertEqual(FakeDet.instance.anchor_prior, FakePre.anchor)
        self.assertEqual(stats, {"matches": 1, "records": 1, "closed": [1]})


if __name__ == "__main__":
    unittest.main()
