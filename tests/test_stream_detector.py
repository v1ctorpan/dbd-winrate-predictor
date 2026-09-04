import csv
import os
import tempfile
import unittest

import cv2
import numpy as np

import make_report
import stream_detector as sd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
GEN = os.path.join(BASE, "picture", "gen.jpg")


def _sorted_frames():
    names = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    return [(cv2.imread(os.path.join(SRC, f)), f) for f in names]


def _pasted_frame(x, y, scale=1.0):
    """在深灰背景固定位置粘贴放大的 gen 模板，制造可检测锚点帧。"""
    tpl = cv2.imread(GEN)
    th, tw = tpl.shape[:2]
    w, h = int(tw * scale), int(th * scale)
    t = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    frame[y:y + h, x:x + w] = t
    return frame


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
            blank = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            r = det.feed(blank, "frame_00_00.0.jpg")
            self.assertIsNone(r)
            self.assertEqual(det.state, "WAIT_ANCHOR")

    def test_wait_requires_stable_position_across_frames(self):
        """菜单误报每帧位置乱跳, 不应触发开局; 稳定同位置连续出现才开。"""
        with tempfile.TemporaryDirectory() as d:
            det = sd.StreamingDetector("BV1Uu8z6eEVM", d, d)
            jitter = [(200, 300), (900, 700), (1500, 400), (300, 1000),
                      (1100, 200), (600, 800)]
            for i, (x, y) in enumerate(jitter):
                det.feed(_pasted_frame(x, y), f"frame_00_0{i}.0.jpg")
            self.assertEqual(det.state, "WAIT_ANCHOR")
            self.assertEqual(det.match_no, 0)

            for i in range(4):
                det.feed(_pasted_frame(400, 400), f"frame_00_0{i + 10}.0.jpg")
            self.assertNotEqual(det.state, "WAIT_ANCHOR")
            self.assertEqual(det.match_no, 1)

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

    def test_hook_slots_recalibrated_after_late_hooks(self):
        """开局 12 帧无人上钩 -> 一次性槽位校准为空; 后续出现上钩竖线时,
        前向滚动重校准应锁定槽位, 使后续帧 hooks>0 (回归: 之前永远 0)。"""
        import hud_regions
        cfg = hud_regions.load_regions(os.path.join(BASE, "config", "hud_regions.json"))
        x, y, scale = 400, 400, 1.0
        anchor = {"x": x, "y": y, "w": 35, "h": 32, "scale": scale}
        resolved = hud_regions.resolve_regions(cfg, anchor)
        b = resolved["hook_p1"]
        c1, c2 = b["x0"] + 3, b["x0"] + 7

        def painted():
            fr = _pasted_frame(x, y, scale)
            fr[b["y0"]:b["y1"], c1] = 255
            fr[b["y0"]:b["y1"], c2] = 255
            return fr

        with tempfile.TemporaryDirectory() as d:
            frames_root = os.path.join(d, "frames")
            report_root = os.path.join(d, "report")
            det = sd.StreamingDetector("BV1Uu8z6eEVM", report_root, frames_root)
            det.budget = 6
            idx = 0
            for _ in range(8):  # WAIT+校准窗口, 无任何上钩线
                det.feed(_pasted_frame(x, y, scale), f"f{idx:05d}.jpg"); idx += 1
            for _ in range(14):  # RECORD, 出现上钩竖线
                det.feed(painted(), f"f{idx:05d}.jpg"); idx += 1
            det.finish()
            csv_path = os.path.join(report_root, "BV1Uu8z6eEVM",
                                    "match_1", "detect_report.csv")
            self.assertTrue(os.path.exists(csv_path), csv_path)
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            hooked_rows = [r for r in rows if r["hooks"].split("/")[0] != "0"]
            self.assertGreater(len(hooked_rows), 0,
                               "late hook lines should eventually be counted")


if __name__ == "__main__":
    unittest.main()
