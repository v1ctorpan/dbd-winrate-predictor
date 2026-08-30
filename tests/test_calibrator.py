import os
import shutil
import sys
import tempfile
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hud_anchor
import calibrator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST1 = os.path.join(BASE, "picture", "test1")
GEN_TPL = os.path.join(BASE, "picture", "gen.jpg")
CFG = os.path.join(BASE, "config", "hud_regions.json")


def candidates_for(frames):
    tpl = cv2.imread(GEN_TPL)
    return [hud_anchor.find_gen_anchors(cv2.imread(os.path.join(TEST1, f)), tpl) for f in frames]


class TestConsensusAnchor(unittest.TestCase):
    def test_finds_true_anchor_across_frames(self):
        cands = candidates_for(["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"])
        anchor = calibrator.consensus_anchor(cands)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 94, delta=3)
        self.assertAlmostEqual(anchor["y"], 536, delta=3)
        self.assertAlmostEqual(anchor["scale"], 1.0, delta=0.1)

    def test_rejects_hudless_frames_only(self):
        cands = candidates_for(["frame_0000.jpg"])
        anchor = calibrator.consensus_anchor(cands)
        self.assertIsNone(anchor)

    def test_picks_highest_score_representative(self):
        cands = candidates_for(["frame_0001.jpg", "frame_0002.jpg"])
        anchor = calibrator.consensus_anchor(cands)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 94, delta=3)
        self.assertAlmostEqual(anchor["y"], 536, delta=3)


class TestCalibrateVideo(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dbd_refs_")
        self.refs_dir = os.path.join(self.tmpdir, "refs", "video_test1")
        os.makedirs(self.refs_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_extracts_healthy_refs_from_opening_frames(self):
        opening = [os.path.join(TEST1, f) for f in
                   ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]]
        anchor = calibrator.calibrate_video(opening, GEN_TPL, CFG, self.refs_dir)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 94, delta=3)
        for i in range(1, 5):
            p = os.path.join(self.refs_dir, f"healthy_p{i}.jpg")
            self.assertTrue(os.path.exists(p), f"missing {p}")
            img = cv2.imread(p)
            self.assertIsNotNone(img)
            self.assertGreater(img.shape[0], 10)
            self.assertGreater(img.shape[1], 10)

    def test_healthy_ref_matches_same_video_face(self):
        opening = [os.path.join(TEST1, f) for f in
                   ["frame_0001.jpg", "frame_0002.jpg"]]
        calibrator.calibrate_video(opening, GEN_TPL, CFG, self.refs_dir)
        import hud_regions
        import state_recognizer
        cfg = hud_regions.load_regions(CFG)
        anchor = {"x": 94, "y": 536, "w": 35, "h": 32, "score": 0.945, "scale": 1.0}
        resolved = hud_regions.resolve_regions(cfg, anchor)
        frame = cv2.imread(os.path.join(TEST1, "frame_0001.jpg"))
        for i in range(1, 5):
            b = resolved[f"survivor_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            ref = cv2.imread(os.path.join(self.refs_dir, f"healthy_p{i}.jpg"))
            self.assertGreater(state_recognizer.ncc(crop, ref), 0.9)


if __name__ == "__main__":
    unittest.main()
