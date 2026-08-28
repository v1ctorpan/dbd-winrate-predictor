import os

import cv2
import numpy as np

import hud_regions

SRC = r"D:\files\code\dbd_pred\picture\test1"
CFG = r"D:\files\code\dbd_pred\config\hud_regions.json"

SLOTS_X = [9, 14]
BG_MARGIN = 12
MIN_RUN = 3

def longest_run(mask):
    best = 0
    cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def line_present(g, colmax, x, thr, w):
    if not (0 <= x < w):
        return False
    window = colmax[max(0, x - 2):min(w, x + 3)]
    return colmax[x] >= window.max() and longest_run(g[:, x] > thr) >= MIN_RUN

def count_hooks(crop, slots_x=SLOTS_X, bg_margin=BG_MARGIN):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    colmax = np.array([float(g[:, x].max()) for x in range(w)], dtype=np.float32)
    bg = float(np.median(colmax))
    thr = bg + bg_margin
    count = 0
    for cx in slots_x:
        found = any(line_present(g, colmax, x, thr, w) for x in (cx - 1, cx, cx + 1))
        if found:
            count += 1
    return count

def hook_crop(frame, resolved, i):
    b = resolved[f"hook_p{i}"]
    return frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]

def main():
    cfg = hud_regions.load_regions(CFG)
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(SRC, fname))
        row = []
        for i in range(1, 5):
            row.append(count_hooks(hook_crop(frame, resolved, i)))
        print(f"{fname}: p1={row[0]} p2={row[1]} p3={row[2]} p4={row[3]}")

if __name__ == "__main__":
    main()
