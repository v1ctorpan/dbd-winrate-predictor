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


if __name__ == "__main__":
    unittest.main()
