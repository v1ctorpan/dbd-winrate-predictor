import argparse
import os
import queue
import threading

import cv2

import dataset_encoder as de
import stream_detector as sd
from extract_frames import frame_name

BASE = os.path.dirname(os.path.abspath(__file__))
PICTURE = os.path.join(BASE, "picture")
DATASET = os.path.join(BASE, "dataset", "videos.jsonl")

_SENTINEL = object()


def _iter_video_frames(video, interval, sample=None):
    """解码线程帧源: 顺序按 interval 取帧, yield (frame, fname)。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    t = 0.0
    count = 0
    while t < duration:
        if sample is not None and count >= sample:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, frame_name(t) + ".jpg"
        t += interval
        count += 1
    cap.release()


def _iter_dir_frames(src_dir, sample=None):
    names = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
    if sample:
        names = names[:sample]
    for n in names:
        img = cv2.imread(os.path.join(src_dir, n))
        if img is not None:
            yield img, n


def _encode_match(bvid, report_root, match_no, videos_path):
    csv_path = os.path.join(report_root, bvid, f"match_{match_no}", "detect_report.csv")
    if not os.path.exists(csv_path):
        return 0
    rec = de.encode_csv(csv_path, f"{bvid}:{match_no}", label=-1)
    return de.append_record(videos_path, rec)


def run_frames_dir(src_dir, bvid, sample=None, videos=DATASET,
                   report_root=None, frames_root=None, budget=12):
    """离线/测试入口: 直接把已有帧目录喂给检测器(单线程串行)。"""
    report_root = report_root or os.path.join(BASE, "report", bvid)
    frames_root = frames_root or PICTURE
    det = sd.StreamingDetector(bvid, report_root, frames_root, hook_names=[bvid])
    det.budget = budget
    closed = []
    for frame, fname in _iter_dir_frames(src_dir, sample=sample):
        r = det.feed(frame, fname)
        if isinstance(r, dict) and "match_end" in r:
            closed.append(r["match_end"])
    closed += det.finish()
    n_rec = 0
    for m in closed:
        n_rec += _encode_match(bvid, report_root, m, videos)
    return {"matches": len(closed), "records": n_rec, "closed": closed}


def run_video(video, bvid, interval=0.5, videos=DATASET, report_root=None,
              frames_root=None, budget=12, sample=None):
    """三线程流水线:
      T1 解码: _iter_video_frames -> q(带 EOF 哨兵)
      T2 检测: 消费 q 喂 StreamingDetector(落盘 match 帧 + 写 CSV) -> closed_q
      T3 编码: 消费 closed_q 里的局号 -> encode+append jsonl
    """
    report_root = report_root or os.path.join(BASE, "report", bvid)
    frames_root = frames_root or PICTURE
    q = queue.Queue(maxsize=64)
    closed_q = queue.Queue()
    errors = []

    def producer():
        try:
            for frame, fname in _iter_video_frames(video, interval, sample=sample):
                q.put((frame, fname))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q.put(_SENTINEL)

    def consumer():
        det = sd.StreamingDetector(bvid, report_root, frames_root, hook_names=[bvid])
        det.budget = budget
        while True:
            item = q.get()
            if item is _SENTINEL:
                q.task_done()
                break
            frame, fname = item
            try:
                r = det.feed(frame, fname)
                if isinstance(r, dict) and "match_end" in r:
                    closed_q.put(r["match_end"])
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            q.task_done()
        for m in det.finish():
            closed_q.put(m)
        closed_q.put(_SENTINEL)

    def encoder():
        while True:
            item = closed_q.get()
            if item is _SENTINEL:
                closed_q.task_done()
                break
            try:
                _encode_match(bvid, report_root, item, videos)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            closed_q.task_done()

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t3 = threading.Thread(target=encoder)
    for t in (t1, t2, t3):
        t.start()
    t1.join()
    t2.join()
    t3.join()
    if errors:
        raise errors[0]
    closed = list(closed_q.queue)
    n_rec = 0
    with open(videos, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_rec += 1
    return {"matches": len(closed), "records": n_rec, "closed": closed}


def main():
    ap = argparse.ArgumentParser(description="DBD 数据产线: 抽帧->流式检测->编码")
    ap.add_argument("source", help="mp4 或 已有帧目录")
    ap.add_argument("bvid")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--videos", default=DATASET)
    ap.add_argument("--budget", type=int, default=12)
    args = ap.parse_args()
    if os.path.isdir(args.source):
        stats = run_frames_dir(args.source, args.bvid, sample=args.sample,
                               videos=args.videos)
    else:
        stats = run_video(args.source, args.bvid, interval=args.interval,
                          videos=args.videos, sample=args.sample,
                          budget=args.budget)
    print(f"matches={stats['matches']} records={stats['records']} -> {args.videos}")
    print(f"closed={stats['closed']}")


if __name__ == "__main__":
    main()
