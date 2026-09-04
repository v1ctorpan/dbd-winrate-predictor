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

    # 区域内(槽位邻域范围内)满足"长亮段"的绝对列集合
    lo = max(b["x0"], 0)
    hi = min(b["x1"], g.shape[1])
    lit = [x for x in range(lo, hi)
           if _is_line(g[b["y0"]:b["y1"], x], thr, min_run, min_top, min_bottom)]
    if not lit:
        return 0
    # 防 overlay: 单一连通亮带若贯穿两槽邻域(如受伤/追击红晕盖过 pip 区) -> 非钩线
    if len(slots) >= 2:
        s0, s1 = sorted(slots)[:2]
        a_hi, b_lo = s0 + tol, s1 - tol
        run_lo = run_hi = lit[0]
        for x in lit[1:]:
            if x == run_hi + 1:
                run_hi = x
            else:
                if run_lo <= a_hi and run_hi >= b_lo:
                    return 0
                run_lo = run_hi = x
        if run_lo <= a_hi and run_hi >= b_lo:
            return 0
    lit_slots = [s for s in slots
                 if any(s - tol <= x <= s + tol for x in lit)]
    return len(lit_slots)


def _is_line(col, thr, min_run, min_top, min_bottom):
    rr = _longest_run_range(col, thr)
    return len(rr) >= min_run and rr[0] <= min_top and rr[-1] >= min_bottom


def count_all(frame, resolved, slots):
    return [count_hooks(frame, resolved, i, slots) for i in range(1, 5)]


def hook_crop(frame, resolved, i):
    b = resolved[f"hook_p{i}"]
    return frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
