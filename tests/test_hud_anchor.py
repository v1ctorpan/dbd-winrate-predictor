import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hud_anchor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIR = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
GEN_TPL = os.path.join(BASE, "picture", "gen.jpg")

# 真实 1080p 视频的 HUD 锚点(开局共识校准得到)
REAL_ANCHOR = (121, 847)
REAL_SCALE = 1.3


def load(fname):
    return cv2.imread(os.path.join(FRAME_DIR, fname))


class TestDetectAnchorWithPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = cv2.imread(GEN_TPL)

    def test_prior_overrides_larger_scale_false_match(self):
        # frame_00_30 曾有 scale1.5@(921,353) 误匹配被优先选中
        frame = load("frame_00_30.0.jpg")
        anchor = hud_anchor.detect_anchor(frame, self.tpl, prior=REAL_ANCHOR)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], REAL_ANCHOR[0], delta=5)
        self.assertAlmostEqual(anchor["y"], REAL_ANCHOR[1], delta=5)
        self.assertAlmostEqual(anchor["scale"], REAL_SCALE, delta=0.15)

    def test_prior_beats_higher_score_false_match(self):
        # frame_02_10 有 scale0.6@(161,674) score0.852 > 真实 0.787
        frame = load("frame_02_10.0.jpg")
        anchor = hud_anchor.detect_anchor(frame, self.tpl, prior=REAL_ANCHOR)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], REAL_ANCHOR[0], delta=5)
        self.assertAlmostEqual(anchor["y"], REAL_ANCHOR[1], delta=5)

    def test_prior_frame_05_10(self):
        # frame_05_10 曾有 scale1.5@(640,24) 误匹配
        frame = load("frame_05_10.0.jpg")
        anchor = hud_anchor.detect_anchor(frame, self.tpl, prior=REAL_ANCHOR)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], REAL_ANCHOR[0], delta=5)
        self.assertAlmostEqual(anchor["y"], REAL_ANCHOR[1], delta=5)

    def test_prior_frame_15_40(self):
        # frame_15_40 曾有 scale1.9@(1390,315) 误匹配
        frame = load("frame_15_40.0.jpg")
        anchor = hud_anchor.detect_anchor(frame, self.tpl, prior=REAL_ANCHOR)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], REAL_ANCHOR[0], delta=5)
        self.assertAlmostEqual(anchor["y"], REAL_ANCHOR[1], delta=5)


class TestDetectAnchorNoPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = cv2.imread(GEN_TPL)

    def test_prefers_highest_score_within_reasonable_scale(self):
        # 无 prior 时选最高分候选(720p test1: scale1.0@(94,536) score0.945 最高)
        t1 = os.path.join(BASE, "picture", "test1", "frame_0001.jpg")
        frame = cv2.imread(t1)
        anchor = hud_anchor.detect_anchor(frame, self.tpl)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 94, delta=3)
        self.assertAlmostEqual(anchor["y"], 536, delta=3)


if __name__ == "__main__":
    unittest.main()
