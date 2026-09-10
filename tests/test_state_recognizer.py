import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_report as mr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIR = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
BV16_DIR = os.path.join(BASE, "picture", "BV16QtT6ZEPq")
REAL_ANCHOR = (121, 847)


def setup_refs():
    cfg = mr.hud_regions.load_regions(mr.CFG)
    tpl = cv2.imread(mr.ANCHOR_TPL)
    names = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg"))
    opening, anch, _ = mr.pick_opening_frame(
        FRAME_DIR, names,
        lambda fr, t: mr.hud_anchor.detect_anchor(fr, t, prior=REAL_ANCHOR),
        tpl, cfg)
    refs = mr.build_refs(anch["scale"])
    refs["healthy"] = mr.build_opening_refs(opening, mr.hud_regions.resolve_regions(cfg, anch))["healthy"]
    return cfg, anch, refs


def crop_slot(frame, cfg, anch, p):
    r = mr.hud_regions.resolve_regions(cfg, anch)
    b = r["survivor_p%d" % p]
    return frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]


class TestRedDiagMeanAlign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.anch, cls.refs = setup_refs()

    def test_injured_red_diag_detected_despite_brightness(self):
        # frame_16_30 p2 用户确认为 injured；红斜线曾被直方图匹配抹平
        frame = cv2.imread(os.path.join(FRAME_DIR, "frame_16_30.0.jpg"))
        crop = crop_slot(frame, self.cfg, self.anch, 2)
        h, w = crop.shape[:2]
        ref = cv2.resize(self.refs["healthy"][1], (w, h), interpolation=cv2.INTER_AREA)
        self.assertGreater(mr.state_recognizer.red_diag_clusters(crop, ref), 0)

    def test_healthy_not_flagged_injured(self):
        # frame_05_00 p3 健康但整体变亮（HUD 高亮），均值对齐后不应有红斜线
        frame = cv2.imread(os.path.join(FRAME_DIR, "frame_05_00.0.jpg"))
        crop = crop_slot(frame, self.cfg, self.anch, 3)
        h, w = crop.shape[:2]
        ref = cv2.resize(self.refs["healthy"][2], (w, h), interpolation=cv2.INTER_AREA)
        self.assertEqual(mr.state_recognizer.red_diag_clusters(crop, ref), 0)

    def test_healthy_brightened_not_flagged(self):
        # frame_08_40 p3 用户确认为 healthy（刚恢复，图标高亮）
        frame = cv2.imread(os.path.join(FRAME_DIR, "frame_08_40.0.jpg"))
        crop = crop_slot(frame, self.cfg, self.anch, 3)
        h, w = crop.shape[:2]
        ref = cv2.resize(self.refs["healthy"][2], (w, h), interpolation=cv2.INTER_AREA)
        self.assertEqual(mr.state_recognizer.red_diag_clusters(crop, ref), 0)

    def test_injured_16_30_p2_classified(self):
        # 用户确认真值：16_30 p2 = injured
        frame = cv2.imread(os.path.join(FRAME_DIR, "frame_16_30.0.jpg"))
        crop = crop_slot(frame, self.cfg, self.anch, 2)
        icons = mr.load_official_icons()
        state = mr.classify(crop, 1, self.refs, icon_tpl=icons)
        self.assertEqual(state, "injured")

    def test_executed_icon_is_loaded(self):
        icons = mr.load_official_icons()
        self.assertIn("executed", icons)
        self.assertIsNotNone(icons["executed"])

    def test_executed_icon_classifies_as_executed(self):
        icons = mr.load_official_icons()
        state = mr.classify(icons["executed"], 0, self.refs, icon_tpl=icons)
        self.assertEqual(state, "executed")


class TestEscapedHighlighted(unittest.TestCase):
    """BV16 末帧 p1 是"刚逃出、图标高亮"的逃跑图标(穿门小人), 与暗色版语义相同,
    应识别为 escaped(否则结局标签漏计)。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = mr.hud_regions.load_regions(mr.CFG)
        cls.anch = {"x": 143, "y": 806, "scale": 1.5}
        cls.resolved = mr.hud_regions.resolve_regions(cls.cfg, cls.anch)
        cls.refs = mr.build_refs(cls.anch["scale"])
        opening = cv2.imread(os.path.join(BV16_DIR, "frame_00_10.0.jpg"))
        cls.refs["healthy"] = mr.build_opening_refs(opening, cls.resolved)["healthy"]
        cls.icons = mr.load_official_icons()

    def _crop(self, name, p):
        frame = cv2.imread(os.path.join(BV16_DIR, name))
        b = self.resolved[f"survivor_p{p}"]
        return frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]

    def test_highlighted_escape_icon_classifies_as_escaped(self):
        crop = self._crop("frame_04_50.0.jpg", 1)
        self.assertEqual(mr.classify(crop, 0, self.refs, self.icons), "escaped")

    def test_normal_escaped_frames_still_escaped(self):
        # p2 为暗黄逃跑图标, 不应因新增高亮模板而回退
        crop = self._crop("frame_04_50.0.jpg", 2)
        self.assertEqual(mr.classify(crop, 1, self.refs, self.icons), "escaped")

    def test_non_escape_slot_not_flagged_escaped(self):
        # p3 不是逃跑图标, 不得误判为 escaped
        crop = self._crop("frame_04_50.0.jpg", 3)
        self.assertNotEqual(mr.classify(crop, 2, self.refs, self.icons), "escaped")


if __name__ == "__main__":
    unittest.main()
