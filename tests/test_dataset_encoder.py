import csv
import json
import os
import tempfile
import unittest

import dataset_encoder as de


class TestParseTime(unittest.TestCase):
    def test_parse_time_integer_seconds(self):
        self.assertEqual(de.parse_time("frame_00_00.0.jpg"), 0)
        self.assertEqual(de.parse_time("frame_00_10.0.jpg"), 20)
        self.assertEqual(de.parse_time("frame_01_30.0.jpg"), 180)   # 90s = 180 half-seconds
        self.assertEqual(de.parse_time("frame_12_45.0.jpg"), 1530)

    def test_parse_time_half_second(self):
        self.assertEqual(de.parse_time("frame_00_00.5.jpg"), 1)     # 0.5s = 1 half-second
        self.assertEqual(de.parse_time("frame_01_30.5.jpg"), 181)
        self.assertEqual(de.parse_time("frame_03_07.5.jpg"), 375)   # 187.5s = 375 half-seconds


class TestOneHotState(unittest.TestCase):
    def test_one_hot(self):
        v = de.one_hot_state("injured")
        self.assertEqual(len(v), 6)
        self.assertEqual(v, [0, 1, 0, 0, 0, 0])

    def test_unknown_maps_to_healthy(self):
        v = de.one_hot_state("unknown")
        self.assertEqual(v[0], 1)

    def test_executed_maps_to_dead_for_dataset(self):
        v = de.one_hot_state("executed")
        self.assertEqual(v, [0, 0, 0, 0, 1, 0])


class TestFeatureVector(unittest.TestCase):
    def test_dimension_30(self):
        row = {"p1": "healthy", "p2": "injured", "p3": "hooked",
               "p4": "dying", "hooks": "1/2/0/1", "gens": "3", "frame": "frame_01_00.0.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(len(v), 30)

    def test_gens_none_to_minus1(self):
        row = {"p1": "healthy", "p2": "healthy", "p3": "healthy",
               "p4": "healthy", "hooks": "0/0/0/0", "gens": "None", "frame": "frame_00_00.0.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(v[28], -1.0)

    def test_time_raw_seconds(self):
        row = {"p1": "healthy", "p2": "healthy", "p3": "healthy",
               "p4": "healthy", "hooks": "0/0/0/0", "gens": "3", "frame": "frame_01_30.5.jpg"}
        v = de.feature_vector(row)
        self.assertEqual(v[29], 90.5)


class TestCompactFrame(unittest.TestCase):
    ROW = {"p1": "healthy", "p2": "injured", "p3": "executed",
           "p4": "escaped", "hooks": "1/2/0/1", "gens": "3",
           "frame": "frame_01_30.5.jpg"}

    def test_compact_frame_is_10_ints(self):
        f = de.compact_frame(self.ROW)
        self.assertEqual(len(f), 10)
        self.assertEqual(f, [0, 1, 4, 5, 1, 2, 0, 1, 3, 181])
        self.assertTrue(all(isinstance(x, int) for x in f))

    def test_unknown_maps_to_healthy_and_gens_none_to_minus1(self):
        row = {"p1": "unknown", "p2": "healthy", "p3": "healthy", "p4": "healthy",
               "hooks": "0/0/0/0", "gens": "None", "frame": "frame_00_00.0.jpg"}
        f = de.compact_frame(row)
        self.assertEqual(f[:4], [0, 0, 0, 0])
        self.assertEqual(f[8], -1)

    def test_frames_to_features_roundtrip(self):
        frames = [de.compact_frame(self.ROW)]
        feats = de.frames_to_features(frames)
        self.assertEqual(feats, [de.feature_vector(self.ROW)])


class TestEncodeVideo(unittest.TestCase):
    def _write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "scale", "p1", "p2", "p3", "p4", "hooks", "gens", "机器标注"])
            w.writerow(["frame_00_00.0.jpg", "1.0", "healthy", "healthy", "healthy", "healthy", "0/0/0/0", "5", "正常"])
            w.writerow(["frame_00_10.0.jpg", "1.0", "injured", "healthy", "healthy", "healthy", "0/0/0/0", "5", "正常"])

    def test_encode_csv_compact(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "in.csv")
            self._write_csv(csv_path)
            rec = de.encode_csv(csv_path, "BV1", label=4)
            self.assertEqual(rec["id"], "BV1")
            self.assertEqual(rec["match"], 1)
            self.assertEqual(rec["title"], "")
            self.assertEqual(rec["url"], "")
            self.assertEqual(rec["label"], 4)
            self.assertNotIn("features", rec)
            self.assertEqual(len(rec["frames"]), 2)
            self.assertEqual(len(rec["frames"][0]), 10)
            self.assertEqual(rec["frames"][0][8], 5)
            self.assertEqual(rec["frames"][0][9], 0)
            self.assertEqual(rec["frames"][1][8], 5)
            self.assertEqual(rec["frames"][1][9], 20)

    def test_encode_csv_meta_match(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "in.csv")
            self._write_csv(csv_path)
            rec = de.encode_csv(csv_path, "BV1X", label=-1,
                                meta={"match": 3, "title": "标题",
                                      "url": "https://www.bilibili.com/video/BV1X"})
            self.assertEqual(rec["id"], "BV1X")
            self.assertEqual(rec["match"], 3)
            self.assertEqual(rec["title"], "标题")
            self.assertEqual(rec["url"], "https://www.bilibili.com/video/BV1X")
            self.assertEqual(rec["label"], -1)


class TestWriteVideosJsonl(unittest.TestCase):
    def test_write_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "videos.jsonl")
            rec1 = {"id": "BV1", "title": "", "url": "", "match": 1,
                    "frames": [[0] * 10, [1] * 10], "label": 3}
            rec2 = {"id": "BV16", "title": "", "url": "", "match": 1,
                    "frames": [[0] * 10], "label": 1}
            de.write_videos_jsonl([rec1, rec2], out_path)
            with open(out_path, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["id"], "BV1")
            self.assertEqual(len(lines[1]["frames"]), 1)
            self.assertEqual(lines[1]["label"], 1)


class TestAppendRecord(unittest.TestCase):
    def test_append_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "videos.jsonl")
            de.append_record(p, {"id": "BV1Z", "title": "", "url": "", "match": 1,
                                 "frames": [[0] * 10], "label": -1})
            de.append_record(p, {"id": "BV1Z", "title": "", "url": "", "match": 2,
                                 "frames": [[1] * 10], "label": -1})
            recs = de.read_records(p)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0]["id"], "BV1Z")
            self.assertEqual(recs[1]["id"], "BV1Z")
            self.assertEqual(recs[1]["match"], 2)
            self.assertEqual(recs[1]["label"], -1)


if __name__ == "__main__":
    unittest.main()
