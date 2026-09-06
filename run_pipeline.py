import argparse
import json
import os
import queue
import threading

import cv2

import dataset_encoder as de
import prescan
import stream_detector as sd
from extract_frames import frame_name

BASE = os.path.dirname(os.path.abspath(__file__))
PICTURE = os.path.join(BASE, "picture")
DATASET = os.path.join(BASE, "dataset", "videos.jsonl")

ANCHOR_MIN_RATIO = 0.5

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


def _encode_match(bvid, report_root, match_no, videos_path, meta=None):
    csv_path = os.path.join(report_root, bvid, f"match_{match_no}", "detect_report.csv")
    if not os.path.exists(csv_path):
        return 0
    meta = dict(meta or {})
    meta["match"] = match_no
    rec = de.encode_csv(csv_path, bvid, label=-1, meta=meta)
    return de.append_record(videos_path, rec)


def _resolve_meta(bvid, title=None, url=None):
    """产线元数据: 有 {raw_videos}/{bvid}.info.json 就自动填 title/webpage_url。

    显式 --title/--url 优先；否则 url 兜底 canonical 视频页。
    """
    meta = {"title": "", "url": f"https://www.bilibili.com/video/{bvid}"}
    info = os.path.join(PICTURE, "raw_videos", f"{bvid}.info.json")
    if os.path.exists(info):
        try:
            with open(info, encoding="utf-8") as f:
                d = json.load(f)
            meta["title"] = d.get("title", "")
            meta["url"] = d.get("webpage_url", "") or meta["url"]
        except (OSError, ValueError):
            pass
    if title is not None:
        meta["title"] = title
    if url is not None:
        meta["url"] = url
    return meta


def run_frames_dir(src_dir, bvid, sample=None, videos=DATASET,
                   report_root=None, frames_root=None, budget=12, meta=None):
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
        n_rec += _encode_match(bvid, report_root, m, videos, meta=meta)
    return {"matches": len(closed), "records": n_rec, "closed": closed}


def run_video(video, bvid, interval=0.5, videos=DATASET, report_root=None,
              frames_root=None, budget=12, sample=None, meta=None):
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
                _encode_match(bvid, report_root, item, videos, meta=meta)
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


def _video_duration(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps else 0.0


def _iter_video_timed(video, interval, sample=None):
    """预扫全量用的带时间戳帧源: yield (frame, fname, t)。"""
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
        yield frame, frame_name(t) + ".jpg", t
        t += interval
        count += 1
    cap.release()


def plan_prescan(pre, duration, anchor_min_ratio=ANCHOR_MIN_RATIO):
    """由预扫结果生成执行计划。

    - anchor_ok: 共识锚点可信(HUD 位置已确认);
    - boundaries: 预扫切局点的秒级时间(丢弃 0 / 越界时间);
    - single: 是否整条视频只有一局;
    - segments: [(start, end)] 粗分段(供报告/排查)。
    """
    anchor = pre.anchor
    anchor_ok = bool(anchor) and pre.anchor_ratio >= anchor_min_ratio
    boundaries = sorted(s.t for i, s in enumerate(pre.samples)
                        if i in pre.boundaries)
    boundaries = [t for t in boundaries if 0.0 < t < duration]
    single = len(boundaries) == 0
    segments = []
    start = 0.0
    for b in boundaries:
        segments.append((start, b))
        start = b
    segments.append((start, duration))
    return {"single": single, "anchor_ok": anchor_ok, "anchor": anchor,
            "anchor_ratio": pre.anchor_ratio, "boundaries": boundaries,
            "segments": segments}


def write_prescan_report(report_root, bvid, pre, plan):
    """落盘预检报告(供人工核对): 锚点 + 多局判定 + 粗分段。"""
    rep_dir = os.path.join(report_root, bvid)
    os.makedirs(rep_dir, exist_ok=True)
    report = {
        "bvid": bvid,
        "anchor": plan["anchor"],
        "anchor_ratio": plan["anchor_ratio"],
        "anchor_ok": plan["anchor_ok"],
        "single": plan["single"],
        "boundaries": [
            {"index": i, "t": s.t, "fname": s.fname,
             "gens": s.gens, "sim_prev": s.sim_prev}
            for i, s in enumerate(pre.samples) if i in pre.boundaries
        ],
        "segments": [{"start": a, "end": b} for a, b in plan["segments"]],
        "n_samples": len(pre.samples),
    }
    path = os.path.join(rep_dir, "prescan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def run_video_prescan(video, bvid, interval=0.5, prescan_interval=prescan.DEFAULT_INTERVAL,
                      videos=DATASET, report_root=None, frames_root=None,
                      budget=12, sample=None, meta=None,
                      anchor_min_ratio=ANCHOR_MIN_RATIO):
    """预扫优先的全量产线:
      1. 10s 预扫确认 HUD 锚点 + 判多局 -> 预检报告 prescan.json;
      2. 锚点可信则带 anchor_prior 直入 RECORD 全量跑(单局关闭自动切局;
         多局按预扫边界 force_new_match 强制切段, 段内 RECORD 0/None 换局仍兜底);
      3. 锚点不可信回退旧版 WAIT 产线 run_video。
    """
    report_root = report_root or os.path.join(BASE, "report", bvid)
    frames_root = frames_root or PICTURE
    duration = _video_duration(video)
    pre = prescan.run_prescan(video, interval=prescan_interval)
    plan = plan_prescan(pre, duration, anchor_min_ratio=anchor_min_ratio)
    report_path = write_prescan_report(report_root, bvid, pre, plan)
    if not plan["anchor_ok"]:
        stats = run_video(video, bvid, interval=interval, videos=videos,
                          report_root=report_root, frames_root=frames_root,
                          budget=budget, sample=sample, meta=meta)
        stats["prescan"] = {"single": plan["single"],
                            "anchor": plan["anchor"],
                            "anchor_ratio": plan["anchor_ratio"]}
        stats["prescan_report"] = report_path
        return stats

    det = sd.StreamingDetector(bvid, report_root, frames_root, hook_names=[bvid],
                               anchor_prior=plan["anchor"],
                               detect_match_end=not plan["single"])
    det.budget = budget
    closed = []
    boundaries = sorted(plan["boundaries"])
    for frame, fname, t in _iter_video_timed(video, interval, sample=sample):
        r = None
        if boundaries and t >= boundaries[0] - 1e-6 and det.state == "RECORD":
            r = det.force_new_match(frame, fname)
            boundaries.pop(0)
        else:
            r = det.feed(frame, fname)
        if isinstance(r, dict) and "match_end" in r:
            closed.append(r["match_end"])
    closed += det.finish()
    n_rec = 0
    for m in closed:
        n_rec += _encode_match(bvid, report_root, m, videos, meta=meta)
    return {"matches": len(closed), "records": n_rec, "closed": closed,
            "prescan": {"single": plan["single"], "anchor": plan["anchor"],
                        "anchor_ratio": plan["anchor_ratio"],
                        "boundaries": plan["boundaries"]},
            "prescan_report": report_path}


def main():
    ap = argparse.ArgumentParser(description="DBD 数据产线: 抽帧->流式检测->编码")
    ap.add_argument("source", help="mp4 或 已有帧目录")
    ap.add_argument("bvid")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--videos", default=DATASET)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--prescan", action="store_true", help="mp4 源: 先 10s 预扫确认锚点/判多局再全量跑")
    ap.add_argument("--prescan-interval", type=float, default=prescan.DEFAULT_INTERVAL)
    ap.add_argument("--title", default=None, help="视频标题(缺省自动读 raw_videos/{bvid}.info.json)")
    ap.add_argument("--url", default=None, help="视频 url(缺省 canonical 或 info.json webpage_url)")
    args = ap.parse_args()
    meta = _resolve_meta(args.bvid, title=args.title, url=args.url)
    if os.path.isdir(args.source):
        stats = run_frames_dir(args.source, args.bvid, sample=args.sample,
                               videos=args.videos, meta=meta)
    elif args.prescan:
        stats = run_video_prescan(args.source, args.bvid, interval=args.interval,
                                  prescan_interval=args.prescan_interval,
                                  videos=args.videos, sample=args.sample,
                                  budget=args.budget, meta=meta)
    else:
        stats = run_video(args.source, args.bvid, interval=args.interval,
                          videos=args.videos, sample=args.sample,
                          budget=args.budget, meta=meta)
    print(f"matches={stats['matches']} records={stats['records']} -> {args.videos}")
    print(f"closed={stats['closed']}")


if __name__ == "__main__":
    main()
