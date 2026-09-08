"""10s 预扫(quick prescan): 粗确认 HUD 锚点 + 判断视频是否含多局。

判定模型: 换局必须同时满足
  A. gens 异常跳变 —— 单局内 gens 只减不增, 识别值高于此前递减基线即异常;
  B. 幸存者头像明显区别 —— 换局前/后 4 个头像裁剪 NCC 均值明显低。
A 且 B 才切局; 只满足一个(如 BV1aat 的 gens 跳回 5 但同一批幸存者)不切。

产出:
  - anchor(共识 HUD 锚点) + anchor_ratio: 用于跑全量前确认 HUD 位置;
  - boundaries(sample 下标): 粗分段的切局点; 段内全量 RECORD 换局仍作兜底。
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

import gens_counter
import hud_anchor
import hud_regions

BASE = os.path.dirname(os.path.abspath(__file__))
ANCHOR_TPL = os.path.join(BASE, "picture", "gen.jpg")
CFG = os.path.join(BASE, "config", "hud_regions.json")
GEN_TPL = gens_counter.GEN_TPL

PORTRAIT_NCC_THR = 0.8
DEFAULT_INTERVAL = 10.0
PORTRAIT_MIN_STD = 8.0
MIN_ANCHOR_SCALE = 0.9


@dataclass
class Sample:
    """预扫单样本。sim_prev: 本样本头像与最近前一含头像样本的 4 头像 NCC 均值。"""
    t: float
    fname: str
    gens: object = None
    portraits: object = None
    sim_prev: float = None
    anchor: dict = None


@dataclass
class PrescanResult:
    anchor: object = None
    anchor_ratio: float = 0.0
    samples: list = None
    boundaries: list = None

    def __post_init__(self):
        if self.samples is None:
            self.samples = []
        if self.boundaries is None:
            self.boundaries = []


SIM_CHANGE_THR = 0.35
MIN_AVATAR_RUN = 2
AVATAR_MERGE_WIN = 3


def _avatar_boundaries(samples, sim_change_thr=SIM_CHANGE_THR,
                       min_run=MIN_AVATAR_RUN, merge_win=AVATAR_MERGE_WIN):
    """头像区骤变切局(旧局结束): 换人/结算使 4 头像 NCC(sim_prev)骤降。

    找连续 >=min_run 个"前后头像差异明显"的样本段, 切在段首(首次骤变)。
    转场常夹杂瞬时回稳(如 BV1QUt766Etg 660s sim=0.96 单点), 相邻骤变段
    若间隔 <= merge_win 个样本视为同一转场, 只取最早段首, 避免多切。
    """
    n = len(samples)
    changed = [False] * n
    for i in range(1, n):
        prev, cur = samples[i - 1], samples[i]
        changed[i] = (prev.portraits is not None and cur.portraits is not None
                      and cur.sim_prev is not None and cur.sim_prev < sim_change_thr)
    runs = []
    j = 1
    while j < n:
        if changed[j]:
            start = j
            while j < n and changed[j]:
                j += 1
            if j - start >= min_run:
                runs.append((start, j - 1))
        else:
            j += 1
    merged = []
    prev_end = -1
    for start, end in runs:
        if merged and start - prev_end <= merge_win:
            prev_end = end
            continue
        merged.append(start)
        prev_end = end
    return merged


def find_boundaries(samples, portrait_thr=PORTRAIT_NCC_THR):
    """扫描样本序列, 返回换局边界所在的 sample 下标(旧局结束/新局首帧之间)。

    两级信号:
    - 头像骤变切局(优先): 幸存者换人/结算 -> 头像区内容骤变并持续, 即旧局结束。
      残局段 gens 恒 0/None、不再回 5(如 BV1QUt766Etg), 旧 gens 规则会漏/滞后,
      此信号在此场景更可靠。
    - gens 跳变切局(回退): 无头像骤变时, gens 异常跳变(高于递减基线或同值
      隔断档后复现) 且头像差异明显才切。仅靠前者会在残局 gens 回 5 的误报
      (如本视频 950s) 处多切, 故有头像骤变边界时以它为准。
    """
    avatar = _avatar_boundaries(samples)
    if avatar:
        return avatar

    last_g = None
    gap_len = 0
    zero_seen = False
    out = []
    for i, s in enumerate(samples):
        g = s.gens
        if g is None or g == 0 or not (1 <= g <= 5):
            if g == 0:
                zero_seen = True
            gap_len += 1
            continue
        ended_gap = zero_seen or gap_len >= 2
        jumped = last_g is not None and (g > last_g or (g == last_g and ended_gap))
        if jumped and s.sim_prev is not None and s.sim_prev < portrait_thr:
            out.append(i)
        last_g = g
        gap_len = 0
        zero_seen = False
    return out


def _fname_time(fname):
    """'frame_MM_SS[.H].jpg' -> 秒。"""
    base = fname.rsplit(".", 1)[0]
    mm_ss = base.split("_")
    m, s = int(mm_ss[-2]), float(mm_ss[-1])
    return float(m * 60 + s)


def iter_sample_frames(source, interval=DEFAULT_INTERVAL, max_frames=None):
    """产出 (t, frame)。目录源按已抽帧名排序取全部; 视频源按 interval 定位抽帧。"""
    if os.path.isdir(source):
        names = sorted(f for f in os.listdir(source) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if max_frames:
            names = names[:max_frames]
        for fname in names:
            img = cv2.imread(os.path.join(source, fname))
            if img is None:
                continue
            yield _fname_time(fname), img
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps else 0.0
    t = 0.0
    n = 0
    while t < duration:
        if max_frames and n >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        yield t, frame
        n += 1
        t += interval
    cap.release()


def _pick_anchor(cands):
    """单样本取一个工作锚点: 优先大尺度(>=MIN_ANCHOR_SCALE)簇中的最高分, 否则最高分。"""
    if not cands:
        return None
    big = [m for m in cands if m["scale"] >= MIN_ANCHOR_SCALE]
    m = max(big if big else cands, key=lambda c: c["score"])
    x0, y0, x1, y1 = m["box"]
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "score": m["score"], "scale": m["scale"]}


def _anchor_key(a):
    return a["x"], a["y"], a["scale"]


def _near(a, b, pos_tol=15, scale_tol=0.25):
    return (abs(a["x"] - b["x"]) <= pos_tol and abs(a["y"] - b["y"]) <= pos_tol
            and abs(a["scale"] - b["scale"]) <= scale_tol)


def _consensus_anchor(anchors):
    """对逐样本锚点做位置聚类, 返回 (代表锚点, 占比)。None 输入跳过。"""
    clusters = []
    for i, a in enumerate(anchors):
        if a is None:
            continue
        for c in clusters:
            if _near(a, c["rep"]):
                c["idx"].append(i)
                if a["score"] > c["rep"]["score"]:
                    c["rep"] = a
                break
        else:
            clusters.append({"rep": a, "idx": [i]})
    if not clusters:
        return None, 0.0
    best = max(clusters, key=lambda c: len(c["idx"]))
    total = len(anchors)
    return best["rep"], (len(best["idx"]) / total if total else 0.0)


def _consensus_candidates(cands_per_frame, pos_tol=15, scale_tol=0.25,
                          min_scale=MIN_ANCHOR_SCALE):
    """对每帧所有候选锚点跨帧聚类, 返回 (代表锚点, 占比)。

    单帧最高分会被某类常驻伪匹配(如 BV1QUt766Etg 左缘 x≈10 scale1.1 高分噪声)
    压过真实 HUD 图标; 改为按"不同帧支持数"选主簇, 真实图标位置稳定、
    支持帧数最多, 占比更鲁棒。每簇记录不同帧下标, 避免同帧多候选重复计数。
    """
    clusters = []
    for fi, clist in enumerate(cands_per_frame):
        for c in clist:
            if c["scale"] < min_scale:
                continue
            a = _as_anchor(c)
            for cl in clusters:
                if _near(a, cl["rep"]):
                    cl["frames"].add(fi)
                    if a["score"] > cl["rep"]["score"]:
                        cl["rep"] = a
                    break
            else:
                clusters.append({"rep": a, "frames": {fi}})
    if not clusters:
        return None, 0.0
    best = max(clusters, key=lambda cl: len(cl["frames"]))
    total = len(cands_per_frame)
    return best["rep"], (len(best["frames"]) / total if total else 0.0)


def _as_anchor(c):
    """把 find_gen_anchors 的候选 {box,scale,score} 归一成 {x,y,w,h,scale,score}。"""
    x0, y0, x1, y1 = c["box"]
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "scale": c["scale"], "score": c["score"]}


def _valid_portrait(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return int(g.size) > 0 and float(g.std()) > PORTRAIT_MIN_STD


def portrait_similarity(crops_a, crops_b):
    """4 头像对 NCC 均值; 任一侧头像缺失则取有效对; 少于 2 对返回 None。"""
    if not crops_a or not crops_b:
        return None
    scores = []
    for a, b in zip(crops_a, crops_b):
        if not _valid_portrait(a) or not _valid_portrait(b):
            continue
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
        ga = ga.ravel() - ga.mean()
        gb = gb.ravel() - gb.mean()
        denom = np.linalg.norm(ga) * np.linalg.norm(gb)
        if denom == 0:
            continue
        scores.append(float((ga @ gb) / denom))
    if len(scores) < 2:
        return None
    return float(np.mean(scores))


def run_prescan(source, interval=DEFAULT_INTERVAL, max_frames=None,
                cfg_path=CFG, tpl_path=ANCHOR_TPL):
    """对视频/抽帧目录做 10s 预扫, 返回 PrescanResult。

    两遍: 第一遍只收集每帧全部候选(不读 gens), 用跨帧支持数选主簇作为共识锚点
    ——单帧最高分会被常驻高分伪匹配(如 BV1QUt766Etg 左缘 scale1.1 噪声)带偏;
    第二遍用"共识锚点附近候选"读 gens/头像, 供多局判定。
    """
    tpl = cv2.imread(tpl_path)
    cfg = hud_regions.load_regions(cfg_path)
    refs = gens_counter.load_digit_refs()
    gen = cv2.imread(GEN_TPL)

    cache_encoded = not os.path.isdir(source)
    sampled_frames = []
    cands_per_frame = []
    for t, frame in iter_sample_frames(source, interval=interval, max_frames=max_frames):
        if cache_encoded:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError(f"cannot cache prescan frame at t={t:.1f}s")
            sampled_frames.append((t, encoded))
        else:
            sampled_frames.append((t, frame))
        cands_per_frame.append(hud_anchor.find_gen_anchors(frame, tpl))
    consensus, ratio = _consensus_candidates(cands_per_frame)

    samples = []
    last_good = None
    last_crops = None
    for (t, cached), cands in zip(sampled_frames, cands_per_frame):
        frame = (cv2.imdecode(cached, cv2.IMREAD_COLOR)
                 if cache_encoded else cached)
        fname = f"t={t:.1f}s"
        gens = None
        portraits = None
        sim_prev = None
        eff = None
        if consensus is not None:
            near = [_as_anchor(c) for c in cands
                    if _near(_as_anchor(c), consensus)]
            if near:
                last_good = max(near, key=lambda a: a["score"])
            eff = last_good
        if eff is not None:
            resolved = hud_regions.resolve_regions(cfg, eff)
            gens = gens_counter.count_gens(frame, resolved, refs, gen=gen, anchor=eff)
            crops = []
            for k in ("survivor_p1", "survivor_p2", "survivor_p3", "survivor_p4"):
                r = resolved[k]
                crop = frame[r["y0"]:r["y1"], r["x0"]:r["x1"]]
                if crop.size == 0 or not _valid_portrait(crop):
                    crops = None
                    break
                crops.append(crop)
            if crops:
                portraits = crops
                if last_crops is not None:
                    sim_prev = portrait_similarity(crops, last_crops)
                last_crops = crops
        samples.append(Sample(t=t, fname=fname, gens=gens, portraits=portraits,
                              sim_prev=sim_prev, anchor=eff))
    return PrescanResult(anchor=consensus, anchor_ratio=ratio, samples=samples,
                         boundaries=find_boundaries(samples))
