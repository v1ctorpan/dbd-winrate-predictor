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
BV16 = os.path.join(BASE, "picture", "BV16QtT6ZEPq")

TEST1_TRUTH = {
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

# BV1 换局前第 1 局真值（frame_10_40 换局）
# 无状态模板识别的权威结果：
# 5: 00_00~01_20, 4: 01_30~03_20, 3: 03_30~05_30, 2: 05_40~07_30,
# 1: 09_10~09_30, 0: 09_40~10_30, 换局 10_40 回 5
# (07_40/08_10 为低分噪声帧，tracker 应沿用前帧 2)
BV1_TRUTH = {
    "frame_00_00": 5, "frame_00_10": 5, "frame_00_20": 5, "frame_00_30": 5,
    "frame_00_40": 5, "frame_00_50": 5, "frame_01_00": 5, "frame_01_10": 5,
    "frame_01_20": 5, "frame_01_30": 4, "frame_01_40": 4, "frame_01_50": 4,
    "frame_02_00": 4, "frame_02_10": 4, "frame_02_20": 4, "frame_02_30": 4,
    "frame_02_40": 4, "frame_02_50": 4, "frame_03_00": 4, "frame_03_10": 4,
    "frame_03_20": 4, "frame_03_30": 3, "frame_03_40": 3, "frame_03_50": 3,
    "frame_04_00": 3, "frame_04_10": 3, "frame_04_20": 3, "frame_04_30": 3,
    "frame_04_40": 3, "frame_04_50": 3, "frame_05_00": 3, "frame_05_10": 3,
    "frame_05_20": 3, "frame_05_30": 3, "frame_05_40": 2, "frame_05_50": 2,
    "frame_06_00": 2, "frame_06_10": 2, "frame_06_20": 2, "frame_06_30": 2,
    "frame_06_40": 2, "frame_06_50": 2, "frame_07_00": 2, "frame_07_10": 2,
    "frame_07_20": 2, "frame_07_30": 2, "frame_07_40": 2, "frame_07_50": 2,
    "frame_08_00": 2, "frame_08_10": 2, "frame_08_20": 2, "frame_08_30": 2,
    "frame_08_40": 2, "frame_08_50": 2, "frame_09_00": 2, "frame_09_10": 1,
    "frame_09_20": 1, "frame_09_30": 1, "frame_09_40": 0, "frame_09_50": 0,
    "frame_10_00": 0, "frame_10_10": 0, "frame_10_20": 0, "frame_10_30": 0,
}
# BV16 真值（无状态模板识别权威结果）
# 5: 00_10~01_30, 4: 01_40, 3: 02_00, 2: 02_20~03_30, 1: 03_40~03_50, 0: 04_00+
# (frame_01_50/02_10/03_00 为过渡帧)
BV16_TRUTH = {
    "frame_00_00": None, "frame_00_10": 5, "frame_00_20": 5, "frame_00_30": 5,
    "frame_01_10": 5, "frame_01_20": 5, "frame_01_30": 5, "frame_01_40": 4,
    "frame_02_00": 3, "frame_02_10": None, "frame_03_20": 2, "frame_03_30": 2,
    "frame_03_40": 1, "frame_03_50": 1, "frame_04_00": 0, "frame_04_10": 0,
    "frame_04_20": 0, "frame_04_30": 0, "frame_04_40": 0, "frame_04_50": 0,
}


def load_resolved(anchor=None):
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    if anchor is None:
        anchor = cfg["anchor"]
    return hud_regions.resolve_regions(cfg, anchor)


def load_tracker():
    refs = gens_counter.load_digit_refs()
    return gens_counter.GensTracker(refs)


class TestGensTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
        cls.refs = gens_counter.load_digit_refs()

    def _track_frames(self, frame_dir, truth, anchor, use_gen=True):
        """顺序处理帧目录中真值里的所有帧，返回 {frame: got}"""
        tracker = gens_counter.GensTracker(self.refs, gen=self.gen)
        results = {}
        for name in sorted(truth):
            path = os.path.join(frame_dir, name + ".0.jpg")
            if not os.path.exists(path):
                path = os.path.join(frame_dir, name + ".jpg")
            frame = cv2.imread(path)
            self.assertIsNotNone(frame, f"frame not loaded: {name}")
            resolved = hud_regions.resolve_regions(
                hud_regions.load_regions(CFG), anchor)
            results[name] = tracker.update(frame, resolved, anchor)
        return results

    def test_test1_all_frames(self):
        results = self._track_frames(SRC, TEST1_TRUTH, {
            "x": 94, "y": 536, "w": 35, "h": 32, "scale": 1.0})
        for name, expected in TEST1_TRUTH.items():
            with self.subTest(frame=name):
                self.assertEqual(results[name], expected, f"{name}: {results[name]} != {expected}")

    def test_bv1_first_match(self):
        ga = {"x": 121, "y": 847, "w": 45, "h": 41, "scale": 1.3}
        results = self._track_frames(BV1, BV1_TRUTH, ga)
        for name, expected in BV1_TRUTH.items():
            with self.subTest(frame=name):
                self.assertEqual(results[name], expected, f"{name}: {results[name]} != {expected}")

    def test_bv16_frames(self):
        ga = {"x": 143, "y": 806, "w": 52, "h": 48, "scale": 1.5}
        results = self._track_frames(BV16, BV16_TRUTH, ga)
        for name, expected in BV16_TRUTH.items():
            with self.subTest(frame=name):
                self.assertEqual(results[name], expected, f"{name}: {results[name]} != {expected}")

    def test_reset_clears_state(self):
        ga = {"x": 121, "y": 847, "w": 45, "h": 41, "scale": 1.3}
        tracker = gens_counter.GensTracker(self.refs, gen=self.gen)
        # 第一帧 5
        f = cv2.imread(os.path.join(BV1, "frame_00_20.0.jpg"))
        resolved = hud_regions.resolve_regions(hud_regions.load_regions(CFG), ga)
        got = tracker.update(f, resolved, ga)
        self.assertEqual(got, 5)
        self.assertIsNotNone(tracker.prev_digit)
        tracker.reset()
        self.assertIsNone(tracker.prev_digit)
        self.assertIsNone(tracker.prev_crop)


if __name__ == "__main__":
    unittest.main()
