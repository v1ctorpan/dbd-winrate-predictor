import csv
import os
import tempfile
import unittest

import cv2

import make_report
import stream_detector as sd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")


def _sorted_frames():
    names = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    return [(cv2.imread(os.path.join(SRC, f)), f) for f in names]


class TestApplyHookCfgMultiVideo(unittest.TestCase):
    def _resolved(self):
        return {f"hook_p{i}": {"x0": 0, "y0": 0, "x1": 10, "y1": 10} for i in range(1, 5)}

    def test_applies_when_name_in_hook_names(self):
        resolved = self._resolved()
        anchor = {"x": 121, "y": 847, "w": 45, "h": 41, "scale": 1.3}
        make_report.apply_hook_cfg(resolved, anchor, hook_names=["BV1Uu8z6eEVM"])
        b = resolved["hook_p1"]
        self.assertNotEqual(b["x0"], 0)
        self.assertEqual(b["y0"], int(round(847 + (-245.3846153846154) * 1.3)))

    def test_ignores_other_videos(self):
        resolved = self._resolved()
        anchor = {"x": 121, "y": 847, "scale": 1.3}
        make_report.apply_hook_cfg(resolved, anchor, hook_names=["BV1Z58J6bEoi"])
        self.assertEqual(resolved["hook_p1"], {"x0": 0, "y0": 0, "x1": 10, "y1": 10})


class TestStreamingDetector(unittest.TestCase):
    def test_wait_anchor_ignores_blank(self):
        with tempfile.TemporaryDirectory() as d:
            det = sd.StreamingDetector("BV1Uu8z6eEVM", d, d)
            import numpy as np
            blank = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            r = det.feed(blank, "frame_00_00.0.jpg")
            self.assertIsNone(r)
            self.assertEqual(det.state, "WAIT_ANCHOR")

    def test_single_match_report_and_finish(self):
        with tempfile.TemporaryDirectory() as d:
            frames_root = os.path.join(d, "frames")
            report_root = os.path.join(d, "report")
            det = sd.StreamingDetector("BV1Uu8z6eEVM", report_root, frames_root,
                                       hook_names=["BV1Uu8z6eEVM", "BV1Z58J6bEoi"])
            frames = _sorted_frames()[:16]
            closed = []
            for frame, fname in frames:
                r = det.feed(frame, fname)
                if isinstance(r, dict) and "match_end" in r:
                    closed.append(r)
            det.finish()
            csv_path = os.path.join(report_root, "BV1Uu8z6eEVM",
                                    "match_1", "detect_report.csv")
            self.assertTrue(os.path.exists(csv_path), csv_path)
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            self.assertEqual(set(rows[0]) & {"p1", "p2", "p3", "p4", "hooks", "gens"},
                             {"p1", "p2", "p3", "p4", "hooks", "gens"})
            match_dir = os.path.join(frames_root, "BV1Uu8z6eEVM", "match_1")
            self.assertTrue(os.path.isdir(match_dir))
            self.assertGreaterEqual(len(os.listdir(match_dir)), 1)


if __name__ == "__main__":
    unittest.main()
