import csv
import json
import os

STATES = ["healthy", "injured", "hooked", "dying", "dead", "escaped"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}


def parse_time(fname):
    """返回以半秒为单位的整数时间。frame_MM_SS.5.jpg = MM*120 + SS*2 + 1。"""
    stem = fname.split(".")[0]              # frame_MM_SS
    parts = stem.split("_")
    mm, ss = int(parts[1]), int(parts[2])
    half = 1 if ".5" in fname else 0
    return mm * 120 + ss * 2 + half


def one_hot_state(state):
    idx = STATE_TO_IDX.get(state, 0)    # unknown -> healthy(0)
    v = [0.0] * len(STATES)
    v[idx] = 1.0
    return v


def feature_vector(row):
    feats = []
    for p in ("p1", "p2", "p3", "p4"):
        feats.extend(one_hot_state(row[p]))
    for h in row["hooks"].split("/"):
        feats.append(float(h))
    gens = row["gens"].strip()
    feats.append(-1.0 if gens in ("None", "") else float(gens))
    feats.append(float(parse_time(row["frame"])) / 2.0)
    return feats


def encode_csv(csv_path, video_id, label):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return {
        "id": video_id,
        "title": "",
        "url": "",
        "features": [feature_vector(r) for r in rows],
        "label": int(label),
    }


def write_videos_jsonl(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def main():
    spec = [
        ("BV1Uu8z6eEVM", "report/BV1Uu8z6eEVM/detect_report.csv", 3),
        ("BV16QtT6ZEPq", "report/BV16QtT6ZEPq/detect_report.csv", 1),
    ]
    records = []
    for video_id, csv_path, label in spec:
        rec = encode_csv(csv_path, video_id, label)
        records.append(rec)
        print(f"[{video_id}] {len(rec['features'])} frames label={rec['label']}")
    n = write_videos_jsonl(records, os.path.join("dataset", "videos.jsonl"))
    print(f"wrote {n} videos -> dataset/videos.jsonl")


if __name__ == "__main__":
    main()
