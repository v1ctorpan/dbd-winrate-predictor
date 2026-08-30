import glob
import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibrator
import hook_counter
import hud_anchor
import hud_regions

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_TPL = os.path.join(BASE, "picture", "gen.jpg")
CFG = os.path.join(BASE, "config", "hud_regions.json")
HOOK_CFG = os.path.join(BASE, "config", "hook_regions.json")

# 用户真值: 每槽位 hook 数量,格式 p1p2p3p4
TRUTH_1080 = {
    "frame_02_10.0.jpg": "1000",
    "frame_02_20.0.jpg": "1000",
    "frame_02_50.0.jpg": "1000",
    "frame_04_40.0.jpg": "1101",
    "frame_05_50.0.jpg": "1201",
    "frame_13_30.0.jpg": "0002",
    "frame_14_20.0.jpg": "0002",
    "frame_05_00.0.jpg": "1101",
}

TRUTH_720 = {
    "frame_0000.jpg": "0000",
    "frame_0001.jpg": "0000",
    "frame_0002.jpg": "0001",
    "frame_0003.jpg": "0011",
    "frame_0004.jpg": "0021",
    "frame_0005.jpg": "0021",
    "frame_0006.jpg": "0121",
    "frame_0007.jpg": "0222",
    "frame_0008.jpg": "0222",
    "frame_0009.jpg": "0222",
    "frame_0010.jpg": "0222",
    "frame_0011.jpg": "0222",
}

REAL_ANCHOR = (121, 847)
ANCHOR_720 = (94, 536)


class _BaseCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = cv2.imread(GEN_TPL)
        cls.cfg = hud_regions.load_regions(CFG)
        cls.hook_cfg = None
        if os.path.exists(HOOK_CFG):
            with open(HOOK_CFG, "r", encoding="utf-8") as f:
                import json
                cls.hook_cfg = json.load(f)

    def _resolved(self, fname):
        frame = cv2.imread(os.path.join(self.FRAME_DIR, fname))
        anchor = self._anchor(fname, frame)
        resolved = hud_regions.resolve_regions(self.cfg, anchor)
        if self.hook_cfg is not None and self.APPLY_HOOK_CFG:
            for name, rel in self.hook_cfg["regions"].items():
                if name in resolved:
                    resolved[name] = hud_regions.rel_to_abs(rel, anchor)
        return frame, resolved


class TestHookCounter1080p(_BaseCounts):
    FRAME_DIR = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
    ANCHOR = REAL_ANCHOR
    SLOTS = [194, 201]
    APPLY_HOOK_CFG = True

    def _anchor(self, fname, frame):
        return hud_anchor.detect_anchor(frame, self.tpl, prior=self.ANCHOR)

    def test_calibrated_slots(self):
        anchor = {"x": 121, "y": 847, "w": 45, "h": 41, "score": 0.8, "scale": 1.3}
        resolved = hud_regions.resolve_regions(self.cfg, anchor)
        if self.hook_cfg is not None and self.APPLY_HOOK_CFG:
            for name, rel in self.hook_cfg["regions"].items():
                if name in resolved:
                    resolved[name] = hud_regions.rel_to_abs(rel, anchor)
        files = sorted(glob.glob(os.path.join(self.FRAME_DIR, "*.jpg")))
        slots = calibrator.calibrate_hook_slots(files, resolved)
        self.assertEqual(len(slots), 2)
        for got, exp in zip(slots, self.SLOTS):
            self.assertLessEqual(abs(got - exp), 1, f"槽位 {got} 应接近 {exp}")

    def test_truth_frames(self):
        for fname, truth in TRUTH_1080.items():
            frame, resolved = self._resolved(fname)
            out = hook_counter.count_all(frame, resolved, self.SLOTS)
            self.assertEqual("".join(map(str, out)), truth, f"{fname} 真值 {truth}")


class TestHookCounter720p(_BaseCounts):
    FRAME_DIR = os.path.join(BASE, "picture", "test1")
    ANCHOR = ANCHOR_720
    SLOTS = [155, 160]
    APPLY_HOOK_CFG = False

    def _anchor(self, fname, frame):
        return self.cfg["anchor"]

    def test_calibrated_slots(self):
        resolved = hud_regions.resolve_regions(self.cfg, self.cfg["anchor"])
        files = sorted(glob.glob(os.path.join(self.FRAME_DIR, "*.jpg")))
        slots = calibrator.calibrate_hook_slots(files, resolved)
        self.assertEqual(len(slots), 2)
        for got, exp in zip(slots, self.SLOTS):
            self.assertLessEqual(abs(got - exp), 1, f"槽位 {got} 应接近 {exp}")

    def test_truth_frames(self):
        for fname, truth in TRUTH_720.items():
            frame, resolved = self._resolved(fname)
            out = hook_counter.count_all(frame, resolved, self.SLOTS)
            self.assertEqual("".join(map(str, out)), truth, f"{fname} 真值 {truth}")


if __name__ == "__main__":
    unittest.main()
