import cv2
import numpy as np

REF_W, REF_H = 35, 32
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
PRIOR_ROI_MARGIN = 96
FAST_SEARCH_MIN_PIXELS = 1_800_000

def _cluster(matches, iou_thr=0.3):
    clusters = []
    for m in matches:
        placed = False
        for c in clusters:
            a = m["box"]
            b = c[0]["box"]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union > iou_thr:
                c.append(m)
                placed = True
                break
        if not placed:
            clusters.append([m])
    return clusters

def _best_per_cluster(matches):
    best = []
    for c in _cluster(matches):
        best.append(max(c, key=lambda m: m["score"]))
    best.sort(key=lambda m: -m["scale"])
    return best

def _fast_full_search(frame, template, scales, min_score, ds=0.4, coarse_thr=0.55,
                      min_scale=0.9):
    """大帧无先验全图搜索: 降采样粗扫候选 + 按位置聚类 + 原生局部精化。

    只保留 scale>=min_scale 的候选——WAIT/预扫等下游会丢弃小尺度菜单噪声,
    它们不代表真实 HUD 发电机图标。与全量多尺度匹配语义一致, 但快得多。
    """
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    small = cv2.resize(frame, (max(1, int(fw * ds)), max(1, int(fh * ds))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    kx, ky = fw / sw, fh / sh
    cands = []
    for s in scales:
        if s < min_scale:
            continue
        w, h = int(tw * s), int(th * s)
        if w >= fw or h >= fh:
            continue
        t = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        ts = cv2.resize(t, (max(1, int(w * ds)), max(1, int(h * ds))),
                        interpolation=cv2.INTER_AREA)
        if ts.shape[0] >= sh or ts.shape[1] >= sw:
            continue
        res = cv2.matchTemplate(small, ts, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, loc = cv2.minMaxLoc(res)
        if maxv >= coarse_thr:
            x, y = int(round(loc[0] * kx)), int(round(loc[1] * ky))
            cands.append({"score": float(maxv), "scale": float(s),
                          "box": (x, y, x + w, y + h)})
    if not cands:
        return []
    reps = _best_per_cluster(cands)
    pad = 24
    matches = []
    for c in reps:
        s = c["scale"]
        w, h = int(tw * s), int(th * s)
        cx, cy = c["box"][0], c["box"][1]
        x0 = max(0, cx - pad); y0 = max(0, cy - pad)
        x1 = min(fw, cx + w + pad); y1 = min(fh, cy + h + pad)
        if x1 - x0 < w or y1 - y0 < h:
            continue
        crop = frame[y0:y1, x0:x1]
        t = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, loc = cv2.minMaxLoc(res)
        if maxv >= min_score:
            x, y = loc[0] + x0, loc[1] + y0
            matches.append({"score": float(maxv), "scale": s,
                            "box": (x, y, x + w, y + h)})
    return _best_per_cluster(matches)

def find_gen_anchors(frame, template, scales=SCALES, min_score=0.70, roi=None):
    if roi is not None:
        x0, y0, x1, y1 = roi
        if x1 <= x0 or y1 <= y0:
            return []
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        found = find_gen_anchors(crop, template, scales=scales, min_score=min_score)
        for m in found:
            bx0, by0, bx1, by1 = m["box"]
            m["box"] = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
        return found
    fh, fw = frame.shape[:2]
    if fw * fh >= FAST_SEARCH_MIN_PIXELS:
        return _fast_full_search(frame, template, scales, min_score)
    th, tw = template.shape[:2]
    matches = []
    for s in scales:
        w, h = int(tw * s), int(th * s)
        t = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        if t.shape[0] >= fh or t.shape[1] >= fw:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv >= min_score:
            x, y = maxloc
            matches.append({
                "score": float(maxv),
                "scale": float(s),
                "box": (x, y, x + w, y + h),
            })
    return _best_per_cluster(matches)

def _prior_roi(frame, prior, margin=PRIOR_ROI_MARGIN):
    fh, fw = frame.shape[:2]
    x0 = max(0, prior[0] - margin)
    y0 = max(0, prior[1] - margin)
    x1 = min(fw, prior[0] + margin)
    y1 = min(fh, prior[1] + margin)
    return (x0, y0, x1, y1)

def detect_anchor(frame, template, prior=None, pos_tol=15, snap=2, **kwargs):
    if prior is not None:
        cands = find_gen_anchors(frame, template, roi=_prior_roi(frame, prior), **kwargs)
    else:
        cands = find_gen_anchors(frame, template, **kwargs)
    if not cands:
        return None
    if prior is not None:
        px, py = prior
        near = [m for m in cands
                if abs(m["box"][0] - px) <= pos_tol and abs(m["box"][1] - py) <= pos_tol]
        if not near:
            return None
        m = max(near, key=lambda m: m["score"])
    else:
        m = max(cands, key=lambda m: m["score"])
    x0, y0, x1, y1 = m["box"]
    # 去抖：检测位置与先验偏移 <= snap 像素时锁定到先验，消除亚像素抖动
    if prior is not None and abs(x0 - px) <= snap and abs(y0 - py) <= snap:
        x0, y0 = px, py
        x1, y1 = px + m["box"][2] - m["box"][0], py + m["box"][3] - m["box"][1]
    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "score": m["score"],
        "scale": m["scale"],
    }

def anchor_scale(anchor):
    return anchor["w"] / REF_W
