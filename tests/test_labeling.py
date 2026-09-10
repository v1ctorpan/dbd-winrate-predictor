import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import labeling


class TestInferLabel(unittest.TestCase):
    def test_counts_escaped_in_last_frames(self):
        rows = [
            ["healthy", "healthy", "healthy", "healthy"],
            ["escaped", "dead", "escaped", "executed"],
        ]
        self.assertEqual(labeling.infer_label(rows), 2)

    def test_escaped_only_before_window_ignored(self):
        rows = [["escaped", "healthy", "healthy", "healthy"]] + \
               [["healthy", "healthy", "healthy", "healthy"]] * 50
        self.assertEqual(labeling.infer_label(rows, window=10), 0)

    def test_transient_highlight_resolves_over_window(self):
        # BV16 型: 末帧 p1 高亮被误判, 但窗口内其它帧出现 escaped 仍应计入
        rows = [["escaped", "healthy", "healthy", "healthy"],
                ["injured", "escaped", "unknown", "escaped"]]
        # p1 escaped 在窗口内出现过 -> 计入; p2/p4 也 escaped -> 共 3
        self.assertEqual(labeling.infer_label(rows, window=40), 3)

    def test_all_unknown_returns_none(self):
        rows = [["unknown"] * 4] * 5
        self.assertIsNone(labeling.infer_label(rows))

    def test_no_escaped_returns_zero(self):
        rows = [["hooked", "dead", "dying", "dead"]] * 5
        self.assertEqual(labeling.infer_label(rows), 0)

    def test_infer_label_from_csv(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "detect_report.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["frame", "p1", "p2", "p3", "p4", "hooks", "gens"])
                w.writerow(["frame_00_00.0", "healthy", "healthy", "healthy", "healthy",
                            "0/0/0/0", "5"])
                w.writerow(["frame_01_00.0", "escaped", "dead", "escaped", "executed",
                            "0/0/0/0", "0"])
            self.assertEqual(labeling.infer_label_from_csv(path), 2)


if __name__ == "__main__":
    unittest.main()
