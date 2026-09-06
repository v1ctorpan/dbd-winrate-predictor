import os
import tempfile
import unittest

import cv2
import numpy as np

import calibrator

REGION = {"x0": 200, "y0": 400, "x1": 250, "y1": 430}


def _resolved():
    return {f"hook_p{i}": dict(REGION) for i in range(1, 5)}


def _frame(lit_cols):
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    for x in lit_cols:
        frame[400:430, x] = 255
    return frame


def _write(frames, names, d):
    paths = []
    for fr, name in zip(frames, names):
        p = os.path.join(d, name)
        cv2.imwrite(p, fr)
        paths.append(p)
    return paths


class TestCalibrateSlotsSupport(unittest.TestCase):
    def test_single_overlay_flash_not_locked(self):
        """仅 1 帧 overlay 宽亮带(跨两槽)不应被锁为槽位; 旧逻辑会因 best 对存在而锁错。"""
        with tempfile.TemporaryDirectory() as d:
            clean = [_frame([]) for _ in range(8)]
            flash = _frame(list(range(208, 217)))
            paths = _write(clean + [flash], [f"f{i:03d}.jpg" for i in range(9)], d)
            got = calibrator.calibrate_hook_slots(paths, _resolved(), min_frames=2)
            self.assertEqual(got, [])

    def test_two_frame_flash_not_locked_either(self):
        """仅 2 帧(小于门限)也不锁。"""
        with tempfile.TemporaryDirectory() as d:
            clean = [_frame([]) for _ in range(6)]
            flash = _frame(list(range(208, 217)))
            paths = _write(clean + [flash, flash],
                           [f"f{i:03d}.jpg" for i in range(8)], d)
            got = calibrator.calibrate_hook_slots(paths, _resolved(), min_frames=3)
            self.assertEqual(got, [])

    def test_sustained_slot_pair_still_locked(self):
        """真实双槽同时亮起持续多帧 -> 照常锁定。"""
        with tempfile.TemporaryDirectory() as d:
            frames = [_frame([209, 210, 214, 215]) for _ in range(6)]
            paths = _write(frames, [f"f{i:03d}.jpg" for i in range(6)], d)
            got = calibrator.calibrate_hook_slots(paths, _resolved(), min_frames=2)
            self.assertEqual(len(got), 2)
            for got_x, exp_x in zip(got, [209, 214]):
                self.assertLessEqual(abs(got_x - exp_x), 1)


if __name__ == "__main__":
    unittest.main()
