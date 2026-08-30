import cv2
import numpy as np

MARGIN = 10
MIN_LINE_RUN = 4
SLOT_TOL = 1
TOP_FRAC = 0.5
BOTTOM_FRAC = 0.4


def _longest_run_range(col, thr):
    best = []
    cur = []
    for y, v in enumerate(col):
        if v > thr:
            cur.append(y)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def count_hooks(frame, resolved, i, slots, margin=MARGIN, min_run=MIN_LINE_RUN,
                tol=SLOT_TOL, top_frac=TOP_FRAC, bottom_frac=BOTTOM_FRAC):
    b = resolved[f"hook_p{i}"]
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = g[b["y0"]:b["y1"], max(0, b["x0"] - 2):min(g.shape[1], b["x1"] + 2)]
    colmax = np.array([float(crop[:, x].max()) for x in range(crop.shape[1])])
    bg = float(np.median(colmax))
    thr = bg + margin
    h = b["y1"] - b["y0"]
    min_top = int(h * top_frac)
    min_bottom = int(h * bottom_frac)
    count = 0
    for s in slots:
        found = False
        for x in range(s - tol, s + tol + 1):
            if b["x0"] <= x < b["x1"]:
                rr = _longest_run_range(g[b["y0"]:b["y1"], x], thr)
                if len(rr) >= min_run and rr[0] <= min_top and rr[-1] >= min_bottom:
                    found = True
                    break
        if found:
            count += 1
    return count


def count_all(frame, resolved, slots):
    return [count_hooks(frame, resolved, i, slots) for i in range(1, 5)]


def hook_crop(frame, resolved, i):
    b = resolved[f"hook_p{i}"]
    return frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
