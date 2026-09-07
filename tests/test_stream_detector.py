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


_SORTED_CACHE = None

def _sorted_frames():
    global _SORTED_CACHE
    if _SORTED_CACHE is None:
        names = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
        _SORTED_CACHE = [(cv2.imread(os.path.join(SRC, f)), f) for f in names]
    return _SORTED_CACHE


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


class _ScriptedGens:
    """按脚本序列返回 gens 的替身，绕过真实 HUD 识别，聚焦换局判定。"""

    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def update(self, frame, resolved, anchor):
        v = self.seq[self.i] if self.i < len(self.seq) else self.seq[-1]
        self.i += 1
        return v

    def reset(self):
        pass


def _drive_to_record(det, frames, cap=40):
    """喂真实 BV1Uu 帧走完 WAIT->CALIBRATE，返回已消耗帧数（此时 state==RECORD）。"""
    for i, (frame, fname) in enumerate(frames[:cap]):
        det.feed(frame, fname)
        if det.state == "RECORD":
            return i + 1
    raise AssertionError("feed-through never reached RECORD state")


class TestAnchorPrior(unittest.TestCase):
    """anchor_prior: 预扫已确认 HUD 锚点 -> 跳过 WAIT 滑动窗口共识直接进 CALIBRATE。"""

    def _detector(self, d, prior=None):
        det = sd.StreamingDetector(
            "BV1Uu8z6eEVM", os.path.join(d, "report"), os.path.join(d, "frames"),
            hook_names=["BV1Uu8z6eEVM", "BV1Z58J6bEoi"], anchor_prior=prior)
        det.budget = 6
        return det

    def test_skips_wait_and_starts_match_on_first_feed(self):
        prior = {"x": 400, "y": 400, "w": 45, "h": 41, "scale": 1.0}
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d, prior)
            det.feed(_pasted_frame(400, 400), "frame_00_00.0.jpg")
            self.assertNotEqual(det.state, "WAIT_ANCHOR")
            self.assertEqual(det.match_no, 1)
            match_dir = os.path.join(d, "frames", "BV1Uu8z6eEVM", "match_1")
            self.assertTrue(os.path.isdir(match_dir))

    def test_reaches_record_without_waiting_for_stable_anchor(self):
        prior = {"x": 400, "y": 400, "w": 45, "h": 41, "scale": 1.0}
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d, prior)
            for i in range(det.budget):
                det.feed(_pasted_frame(400, 400), f"frame_00_0{i}.0.jpg")
            self.assertEqual(det.state, "RECORD")
            self.assertEqual(det.match_no, 1)

    def test_records_gens_with_scripted_value(self):
        prior = {"x": 400, "y": 400, "w": 45, "h": 41, "scale": 1.0}
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d, prior)
            det._gens_tracker = _ScriptedGens([5] * 20)
            for i in range(det.budget):
                det.feed(_pasted_frame(400, 400), f"frame_00_0{i}.0.jpg")
            r = det.feed(_pasted_frame(400, 400), "frame_00_10.0.jpg")
            self.assertIsInstance(r, dict)
            self.assertEqual(r["gens"], 5)


class TestMatchEndRegression(unittest.TestCase):
    """回归: BV1aatX6uE3C 误切成 4 段的根因是"孤立单帧 0 后紧跟 5"
    (26.5/58.5/175.5s) 触发换局; 修复后连续 0/None 段(>=MIN_END_ZERO_RUN)才换局。"""

    def _detector(self, d):
        return sd.StreamingDetector(
            "BV1Uu8z6eEVM", os.path.join(d, "report"), os.path.join(d, "frames"),
            hook_names=["BV1Uu8z6eEVM", "BV1Z58J6bEoi"])

    def _feed_scripted(self, det, frames, seq, n):
        det._gens_tracker = _ScriptedGens(seq)
        events = []
        for frame, fname in frames[n:n + len(seq)]:
            r = det.feed(frame, fname)
            if isinstance(r, dict) and "match_end" in r:
                events.append(r)
        return events

    def test_isolated_single_zero_does_not_end_match(self):
        """孤立单帧 0(...5,0,5...) 是 HUD 图标瞬时误读, 不得切局。"""
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d)
            frames = _sorted_frames()
            n = _drive_to_record(det, frames)
            events = self._feed_scripted(det, frames, [5, 5, 0, 5, 5, 5], n)
            det.finish()
            self.assertEqual(events, [], "isolated single 0 must not end match")
            csv_path = os.path.join(d, "report", "BV1Uu8z6eEVM",
                                    "match_1", "detect_report.csv")
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            gs = [r["gens"] for r in rows]
            self.assertIn("0", gs, "isolated 0 frame must still be recorded")
            self.assertEqual(gs[-1], "5", "later frames stay in same match")
            self.assertEqual(det.match_no, 1)

    def test_sustained_zero_run_still_ends_match(self):
        """连续 >=2 帧 0/None 后回 5 才是真换局(BV1Uu frame_10_40 前 09_40 起
        持续 0)。修复不得把真换局一起压制。"""
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d)
            frames = _sorted_frames()
            n = _drive_to_record(det, frames)
            events = self._feed_scripted(det, frames, [5, 5, 0, 0, 5], n)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["match_end"], 1)
            self.assertEqual(det.state, "CALIBRATE")
            self.assertEqual(det.match_no, 2)


class TestPrescanDrivenEnd(unittest.TestCase):
    """预扫集成所需的检测器能力:
       - detect_match_end=False: 预扫判单局 -> 禁用 0/None 后回 5 的自动切局;
       - force_new_match: 预扫判多局 -> 在预扫边界强制开新局(即使无死段)。"""

    def _detector(self, d, detect_match_end=True):
        return sd.StreamingDetector(
            "BV1Uu8z6eEVM", os.path.join(d, "report"), os.path.join(d, "frames"),
            hook_names=["BV1Uu8z6eEVM", "BV1Z58J6bEoi"],
            detect_match_end=detect_match_end)

    def _drive(self, det):
        frames = _sorted_frames()
        n = _drive_to_record(det, frames)
        return frames, n

    def _feed_scripted(self, det, frames, seq, n):
        det._gens_tracker = _ScriptedGens(seq)
        events = []
        for frame, fname in frames[n:n + len(seq)]:
            r = det.feed(frame, fname)
            if isinstance(r, dict) and "match_end" in r:
                events.append(r)
        return events

    def test_detect_match_end_false_suppresses_sustained_zero_run(self):
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d, detect_match_end=False)
            frames, n = self._drive(det)
            events = self._feed_scripted(det, frames, [5, 5, 0, 0, 0, 5], n)
            det.finish()
            self.assertEqual(events, [])
            self.assertEqual(det.match_no, 1)
            match1 = os.path.join(d, "report", "BV1Uu8z6eEVM", "match_1")
            match2 = os.path.join(d, "report", "BV1Uu8z6eEVM", "match_2")
            self.assertTrue(os.path.exists(os.path.join(match1, "detect_report.csv")))
            self.assertFalse(os.path.exists(match2))

    def test_force_new_match_splits_even_without_zero_run(self):
        """预扫在"末局速杀 5->5 无死段"等情形仍要切局 -> 显式强制开新局。"""
        with tempfile.TemporaryDirectory() as d:
            det = self._detector(d)
            frames, n = self._drive(det)
            det._gens_tracker = _ScriptedGens([5] * 40)
            frame, fname = frames[n]
            r = det.force_new_match(frame, fname)
            self.assertEqual(r["match_end"], 1)
            self.assertEqual(det.state, "CALIBRATE")
            self.assertEqual(det.match_no, 2)
            for i in range(1, det.budget + 2):
                det.feed(frames[n + i][0], frames[n + i][1])
            det.finish()
            match2 = os.path.join(d, "report", "BV1Uu8z6eEVM", "match_2")
            self.assertTrue(os.path.exists(os.path.join(match2, "detect_report.csv")))


if __name__ == "__main__":
    unittest.main()
