import csv
import json
import os

STATES = ["healthy", "injured", "hooked", "dying", "dead", "escaped"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
STATE_TO_IDX["executed"] = STATE_TO_IDX["dead"]


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


def compact_frame(row, t0=0):
    """把一行检测结果压成 10 个整数：
    [p1..p4 状态类别索引, hooks×4, gens(-1..5), 半秒时间]。
    占用远小于 30 维浮点 one-hot；unknown->healthy(0)、executed->dead(4)。
    t0 = 该局首帧的半秒时间，用于把时间归零（默认 0 = 绝对时间）。"""
    states = [STATE_TO_IDX.get(row[p], 0) for p in ("p1", "p2", "p3", "p4")]
    hooks = [int(h) for h in row["hooks"].split("/")]
    gens = row["gens"].strip()
    gens = -1 if gens in ("None", "") else int(gens)
    return states + hooks + [gens, parse_time(row["frame"]) - t0]


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


def frames_to_features(frames):
    """把紧凑帧元组还原成 30 维特征向量列表（供模型/兼容旧格式）。"""
    out = []
    for f in frames:
        feats = []
        for i in range(4):
            v = [0.0] * len(STATES)
            v[int(f[i])] = 1.0
            feats.extend(v)
        feats.extend(float(h) for h in f[4:8])
        feats.append(float(f[8]))
        feats.append(f[9] / 2.0)
        out.append(feats)
    return out


def encode_csv(csv_path, video_id, label, meta=None):
    """把 detect_report.csv 编码为一条 dataset 记录（紧凑帧元组）。"""
    meta = meta or {}
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    t0 = parse_time(rows[0]["frame"]) if rows else 0
    # HUD 短暂消失(gens=None)时复用前一有效值；局首缺失保持 -1（未知）。
    frames = []
    carry = None
    for r in rows:
        raw = r["gens"].strip()
        g = None if raw in ("None", "") else int(raw)
        if g is None:
            g = carry
        else:
            carry = g
        r2 = dict(r)
        r2["gens"] = "" if g is None else str(g)
        frames.append(compact_frame(r2, t0=t0))
    return {
        "id": video_id,
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "match": int(meta.get("match", 1)),
        "frames": frames,
        "label": int(label),
    }


def write_videos_jsonl(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def append_record(videos_path, record):
    os.makedirs(os.path.dirname(videos_path), exist_ok=True)
    with open(videos_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(videos_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_records(videos_path):
    with open(videos_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    spec = [
        ("BV1Uu8z6eEVM", "report/BV1Uu8z6eEVM/detect_report.csv", 3),
        ("BV16QtT6ZEPq", "report/BV16QtT6ZEPq/detect_report.csv", 1),
    ]
    records = []
    for video_id, csv_path, label in spec:
        rec = encode_csv(csv_path, video_id, label)
        records.append(rec)
        print(f"[{video_id}] {len(rec['frames'])} frames label={rec['label']}")
    n = write_videos_jsonl(records, os.path.join("dataset", "videos.jsonl"))
    print(f"wrote {n} videos -> dataset/videos.jsonl")


if __name__ == "__main__":
    main()
