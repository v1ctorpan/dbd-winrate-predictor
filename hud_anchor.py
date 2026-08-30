import cv2
import numpy as np

REF_W, REF_H = 35, 32
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

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

def find_gen_anchors(frame, template, scales=SCALES, min_score=0.70):
    fh, fw = frame.shape[:2]
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
    if not matches:
        return []

    best_per_cluster = []
    for c in _cluster(matches):
        best = max(c, key=lambda m: m["score"])
        best_per_cluster.append(best)
    best_per_cluster.sort(key=lambda m: -m["scale"])
    return best_per_cluster

def detect_anchor(frame, template, prior=None, pos_tol=15, **kwargs):
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
