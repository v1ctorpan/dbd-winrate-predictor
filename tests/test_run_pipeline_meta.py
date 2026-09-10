import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import run_pipeline as rp


class TestResolveMeta(unittest.TestCase):
    def test_without_info_json_uses_canonical_url(self):
        meta = rp._resolve_meta("BV1X")
        self.assertEqual(meta["title"], "")
        self.assertEqual(meta["url"], "https://www.bilibili.com/video/BV1X")

    def test_reads_info_json_title_and_url(self):
        with tempfile.TemporaryDirectory() as d:
            info = {"title": "标题X", "webpage_url": "https://www.bilibili.com/video/BV1X"}
            raw = os.path.join(d, "raw_videos")
            os.makedirs(raw)
            with open(os.path.join(raw, "BV1X.info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f)
            with mock.patch.object(rp, "PICTURE", d):
                meta = rp._resolve_meta("BV1X")
            self.assertEqual(meta["title"], "标题X")
            self.assertEqual(meta["url"], "https://www.bilibili.com/video/BV1X")

    def test_explicit_title_url_override(self):
        with tempfile.TemporaryDirectory() as d:
            info = {"title": "旧标题", "webpage_url": "https://www.bilibili.com/video/BV1X"}
            raw = os.path.join(d, "raw_videos")
            os.makedirs(raw)
            with open(os.path.join(raw, "BV1X.info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f)
            with mock.patch.object(rp, "PICTURE", d):
                meta = rp._resolve_meta("BV1X", title="新标题",
                                        url="https://www.bilibili.com/video/BV1X")
            self.assertEqual(meta["title"], "新标题")


class TestEncodeMatch(unittest.TestCase):
    def _write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "scale", "p1", "p2", "p3", "p4", "hooks", "gens", "机器标注"])
            w.writerow(["frame_00_00.0.jpg", "1.0", "healthy", "healthy", "healthy", "healthy",
                        "0/0/0/0", "5", "正常"])
            w.writerow(["frame_00_10.0.jpg", "1.0", "injured", "healthy", "healthy", "healthy",
                        "0/0/0/0", "5", "正常"])

    def test_encode_match_id_is_pure_bvid(self):
        with tempfile.TemporaryDirectory() as d:
            report_root = os.path.join(d, "report")
            os.makedirs(os.path.join(report_root, "BV1X", "match_3"))
            csv_path = os.path.join(report_root, "BV1X", "match_3", "detect_report.csv")
            self._write_csv(csv_path)
            videos = os.path.join(d, "videos.jsonl")
            meta = {"title": "标题X", "url": "https://www.bilibili.com/video/BV1X"}
            appended = rp._encode_match("BV1X", report_root, 3, videos, meta=meta)
            self.assertEqual(appended, 1)
            with open(videos, encoding="utf-8") as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["id"], "BV1X")
            self.assertEqual(rec["match"], 3)
            self.assertEqual(rec["title"], "标题X")
            self.assertEqual(rec["url"], "https://www.bilibili.com/video/BV1X")
            self.assertEqual(rec["label"], 0)
            self.assertEqual(len(rec["frames"]), 2)
            self.assertEqual(len(rec["frames"][0]), 10)

    def test_encode_match_no_csv_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            n = rp._encode_match("BV1X", os.path.join(d, "report"), 1,
                                 os.path.join(d, "videos.jsonl"))
            self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
