import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hud_regions
import gens_counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(BASE, "config", "hud_regions.json")
SRC = os.path.join(BASE, "picture", "test1")
BV1 = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")

TRUTH = {
    "frame_0000": None,
    "frame_0001": 5,
    "frame_0002": 4,
    "frame_0003": 4,
    "frame_0004": 4,
    "frame_0005": 4,
    "frame_0006": 3,
    "frame_0007": 2,
    "frame_0008": 2,
    "frame_0009": 2,
    "frame_0010": 0,
    "frame_0011": 0,
}


def load_resolved():
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    return hud_regions.resolve_regions(cfg, cfg["anchor"])


class TestGensCounter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolved = load_resolved()
        cls.refs = gens_counter.build_digit_refs()

    def test_all_frames_match_truth(self):
        for name, expected in TRUTH.items():
            frame = cv2.imread(os.path.join(SRC, name + ".jpg"))
            self.assertIsNotNone(frame, f"frame not loaded: {name}")
            with self.subTest(frame=name):
                got = gens_counter.count_gens(frame, self.resolved, self.refs)
                self.assertEqual(got, expected)

    def test_bv1_video_refs(self):
        ga = {"x": 121, "y": 847, "w": 45, "h": 41, "scale": 1.3}
        resolved = hud_regions.resolve_regions(hud_regions.load_regions(CFG), ga)
        refs = gens_counter.build_digit_refs_from_video(
            BV1, resolved,
            {5: ["frame_00_20", "frame_00_30", "frame_11_00"],
             4: ["frame_01_30", "frame_03_20"],
             3: ["frame_03_30", "frame_13_30"],
             2: ["frame_05_40", "frame_14_00"],
             1: ["frame_09_10", "frame_15_30"]}, 1.3)
        gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
        checks = {
            "frame_00_30.0.jpg": 5, "frame_02_10.0.jpg": 4,
            "frame_04_40.0.jpg": 3, "frame_05_50.0.jpg": 2,
            "frame_09_10.0.jpg": 1, "frame_15_40.0.jpg": 1,
        }
        for fname, expected in checks.items():
            with self.subTest(frame=fname):
                frame = cv2.imread(os.path.join(BV1, fname))
                got = gens_counter.count_gens(frame, resolved, refs, gen=gen, anchor=ga)
                self.assertEqual(got, expected)


class TestGensCounterOutOfFrame(unittest.TestCase):
    """回归: BV1QUt766Etg 预扫中 find_gen_anchors 伪检出左上角锚点 (10,178)@1.1,
    gens_row 相对坐标 x0=-29 换算得负 x, numpy 切片返回空 crop 使 cvtColor 崩溃。
    越界/空 crop 应判 None(无 HUD)而不是抛异常。"""

    def _frame(self):
        frame = cv2.imread(os.path.join(BV1, "frame_00_30.0.jpg"))
        self.assertIsNotNone(frame)
        return frame

    def _oob_resolved(self):
        return {"gens_row": {"x0": -22, "y0": 150, "x1": 52, "y1": 190}}

    def test_count_gens_oob_returns_none(self):
        got = gens_counter.count_gens(
            self._frame(), self._oob_resolved(), self.refs_or_refs(), anchor={"scale": 1.1})
        self.assertIsNone(got)

    def test_tracker_oob_returns_none(self):
        tracker = gens_counter.GensTracker(self.refs_or_refs())
        got = tracker.update(self._frame(), self._oob_resolved(), {"scale": 1.1})
        self.assertIsNone(got)

    @staticmethod
    def refs_or_refs():
        return gens_counter.load_digit_refs()


if __name__ == "__main__":
    unittest.main()
