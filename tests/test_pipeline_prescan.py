import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prescan
import run_pipeline as rp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_pre(anchor, ratio, n=100, interval=10.0, boundary_idx=()):
    samples = [prescan.Sample(t=i * interval, fname=f"frame_{i}", gens=None,
                              portraits=None) for i in range(n)]
    pre = prescan.PrescanResult(anchor=anchor, anchor_ratio=ratio, samples=samples,
                                boundaries=list(boundary_idx))
    return pre


class TestPlanPrescan(unittest.TestCase):
    def test_single_match_plan(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91)
        plan = rp.plan_prescan(pre, 1000.0)
        self.assertTrue(plan["single"])
        self.assertTrue(plan["anchor_ok"])
        self.assertEqual(plan["boundaries"], [])
        self.assertEqual(plan["segments"], [(0.0, 1000.0)])

    def test_two_match_plan(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91,
                        boundary_idx=[64])
        plan = rp.plan_prescan(pre, 700.0)
        self.assertFalse(plan["single"])
        self.assertEqual(plan["boundaries"], [640.0])
        self.assertEqual(plan["segments"], [(0.0, 640.0), (640.0, 700.0)])

    def test_boundary_at_video_end_is_dropped(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91,
                        boundary_idx=[64])
        plan = rp.plan_prescan(pre, 640.0)
        self.assertTrue(plan["single"])
        self.assertEqual(plan["boundaries"], [])
        self.assertEqual(plan["segments"], [(0.0, 640.0)])

    def test_low_anchor_ratio_not_ok(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.1)
        plan = rp.plan_prescan(pre, 1000.0)
        self.assertFalse(plan["anchor_ok"])


class FakeDet:
    instances = []

    def __init__(self, bvid, report_root=None, frames_root=None, hook_names=None,
                 anchor_prior=None, detect_match_end=True, **kw):
        self.bvid = bvid
        self.anchor_prior = anchor_prior
        self.detect_match_end = detect_match_end
        self.state = "RECORD"
        self.match_no = 1
        self.force_calls = []
        self.feed_calls = []
        FakeDet.instances.append(self)

    def feed(self, frame, fname):
        self.feed_calls.append(fname)
        return None

    def force_new_match(self, frame, fname):
        self.force_calls.append(fname)
        self.match_no += 1
        return {"match_end": self.match_no - 1}

    def finish(self):
        self.match_no += 1
        return [self.match_no - 1]


def _timed_frames(t0, t1, step):
    """返回 (frame, fname, t) 列表; fname 与产线 frame_name 同构。"""
    from extract_frames import frame_name
    out = []
    t = t0
    while t < t1:
        out.append((None, frame_name(t) + ".jpg", t))
        t += step
    return out


class TestRunVideoPrescan(unittest.TestCase):
    def setUp(self):
        FakeDet.instances = []
        self.frames = _timed_frames(0.0, 20.0, 0.5)

    def _run(self, pre):
        p = mock.patch.object(rp.prescan, "run_prescan", return_value=pre)
        p.start()
        self.addCleanup(p.stop)
        p1 = mock.patch.object(rp, "_video_duration", return_value=20.0)
        p1.start()
        self.addCleanup(p1.stop)
        p2 = mock.patch.object(rp, "_iter_video_timed", return_value=iter(self.frames))
        p2.start()
        self.addCleanup(p2.stop)
        p4 = mock.patch.object(rp.sd, "StreamingDetector", FakeDet)
        p4.start()
        self.addCleanup(p4.stop)
        encoded = []
        p3 = mock.patch.object(rp, "_encode_match",
                               side_effect=lambda *a, **k: (encoded.append(a[2]) or 1))
        p3.start()
        self.addCleanup(p3.stop)
        with tempfile.TemporaryDirectory() as d:
            stats = rp.run_video_prescan("dummy.mp4", "BV1X", report_root=d,
                                         frames_root=os.path.join(d, "frames"),
                                         interval=0.5)
        return stats, encoded

    def test_multi_match_single_detector_with_prior_and_force(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91,
                        n=2, boundary_idx=[1])
        stats, encoded = self._run(pre)
        self.assertEqual(len(FakeDet.instances), 1)
        det = FakeDet.instances[0]
        self.assertEqual(det.anchor_prior["x"], 121)
        self.assertTrue(det.detect_match_end)
        self.assertEqual(det.force_calls, ["frame_00_10.0.jpg"])
        self.assertEqual(stats["matches"], 2)
        self.assertEqual(len(encoded), 2)
        self.assertIn(1, encoded)
        self.assertIn(2, encoded)

    def test_single_match_detect_match_end_disabled(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91, n=2)
        stats, encoded = self._run(pre)
        det = FakeDet.instances[0]
        self.assertFalse(det.detect_match_end)
        self.assertEqual(det.force_calls, [])
        self.assertEqual(stats["matches"], 1)

    def test_report_written(self):
        pre = _fake_pre({"x": 121, "y": 847, "scale": 1.3}, 0.91,
                        n=2, boundary_idx=[1])
        p = mock.patch.object(rp.prescan, "run_prescan", return_value=pre)
        p.start()
        self.addCleanup(p.stop)
        with mock.patch.object(rp, "_video_duration", return_value=20.0), \
                mock.patch.object(rp, "_iter_video_timed",
                                  return_value=iter(self.frames)), \
                mock.patch.object(rp.sd, "StreamingDetector", FakeDet), \
                mock.patch.object(rp, "_encode_match", return_value=1):
            with tempfile.TemporaryDirectory() as d:
                rp.run_video_prescan("dummy.mp4", "BV1X", report_root=d,
                                     frames_root=os.path.join(d, "frames"))
                path = os.path.join(d, "BV1X", "prescan.json")
                self.assertTrue(os.path.exists(path))
                with open(path, encoding="utf-8") as f:
                    rep = json.load(f)
        self.assertEqual(rep["anchor"]["x"], 121)
        self.assertFalse(rep["single"])
        self.assertEqual([b["t"] for b in rep["boundaries"]], [10.0])

    def test_anchor_not_ok_falls_back_to_legacy(self):
        pre = _fake_pre(None, 0.0, n=2)
        p = mock.patch.object(rp.prescan, "run_prescan", return_value=pre)
        p.start()
        self.addCleanup(p.stop)
        with mock.patch.object(rp, "_video_duration", return_value=20.0), \
                mock.patch.object(rp, "run_video", return_value={"matches": 3,
                                                                 "records": 3}) as rv:
            stats = rp.run_video_prescan("dummy.mp4", "BV1X", report_root="/tmp/x",
                                         frames_root="/tmp/f")
        rv.assert_called_once()
        self.assertEqual(stats["matches"], 3)


if __name__ == "__main__":
    unittest.main()
