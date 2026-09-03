import json
import os
import tempfile
import unittest

import run_pipeline as rp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")


class TestPipeline(unittest.TestCase):
    def test_frames_dir_pipeline_writes_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            videos = os.path.join(d, "videos.jsonl")
            report_root = os.path.join(d, "report")
            frames_root = os.path.join(d, "frames")
            stats = rp.run_frames_dir(SRC, "BV1Uu8z6eEVM", sample=16,
                                      videos=videos, report_root=report_root,
                                      frames_root=frames_root)
            self.assertGreaterEqual(stats["matches"], 1)
            with open(videos, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertGreaterEqual(len(lines), 1)
            self.assertTrue(lines[0]["id"].startswith("BV1Uu8z6eEVM:"))
            self.assertEqual(len(lines[0]["features"][0]), 30)


if __name__ == "__main__":
    unittest.main()
