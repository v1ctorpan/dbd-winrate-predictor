import os

import cv2
import numpy as np

import hook_counter
import hud_anchor
import hud_regions


def consensus_anchor(cands_per_frame, pos_tol=15, min_frames=2):
    clusters = []
    for fi, cands in enumerate(cands_per_frame):
        for c in cands:
            x, y = c["box"][0], c["box"][1]
            placed = False
            for cl in clusters:
                if abs(cl["x"] - x) <= pos_tol and abs(cl["y"] - y) <= pos_tol:
                    cl["members"].append((fi, c))
                    cl["frames"].add(fi)
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "x": x,
                    "y": y,
                    "members": [(fi, c)],
                    "frames": {fi},
                })
    if not clusters:
        return None
    best = max(clusters, key=lambda cl: (len(cl["frames"]),
                                         max(m[1]["score"] for m in cl["members"])))
    if len(best["frames"]) < min_frames:
        return None
    rep = max(best["members"], key=lambda m: m[1]["score"])[1]
    x0, y0, x1, y1 = rep["box"]
    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "score": rep["score"],
        "scale": rep["scale"],
    }


def calibrate_video(frame_paths, gen_tpl_path, cfg_path, refs_dir, pos_tol=15, min_frames=2):
    tpl = cv2.imread(gen_tpl_path)
    cfg = hud_regions.load_regions(cfg_path)
    cands = [hud_anchor.find_gen_anchors(cv2.imread(p), tpl) for p in frame_paths]
    anchor = consensus_anchor(cands, pos_tol=pos_tol, min_frames=min_frames)
    if anchor is None:
        return None
    resolved = hud_regions.resolve_regions(cfg, anchor)

    best_fi, best_score = -1, -1.0
    for fi, cand_list in enumerate(cands):
        for c in cand_list:
            if (abs(c["box"][0] - anchor["x"]) <= pos_tol
                    and abs(c["box"][1] - anchor["y"]) <= pos_tol
                    and c["score"] > best_score):
                best_score = c["score"]
                best_fi = fi

    os.makedirs(refs_dir, exist_ok=True)
    frame = cv2.imread(frame_paths[best_fi])
    for i in range(1, 5):
        b = resolved[f"survivor_p{i}"]
        crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
        cv2.imwrite(os.path.join(refs_dir, f"healthy_p{i}.jpg"), crop)
    return anchor


def calibrate_hook_slots(frame_paths, resolved, margin=hook_counter.MARGIN,
                         min_run=hook_counter.MIN_LINE_RUN,
                         top_frac=hook_counter.TOP_FRAC, bottom_frac=hook_counter.BOTTOM_FRAC,
                         min_gap=4, max_gap=12):
    """通过"同帧共现"找槽位: 真实钩线成对出现(两槽位常同时亮起),
    而区域内的静态装饰线(如头像轮廓)独立出现, 参与共现次数低, 会被排除。"""
    cols_by_item = {}
    for i in range(1, 5):
        b = resolved[f"hook_p{i}"]
        h = b["y1"] - b["y0"]
        min_top = int(h * top_frac)
        min_bottom = int(h * bottom_frac)
        for fp in frame_paths:
            g = cv2.imread(fp)
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            crop = g[b["y0"]:b["y1"], max(0, b["x0"] - 2):min(g.shape[1], b["x1"] + 2)]
            colmax = np.array([float(crop[:, x].max()) for x in range(crop.shape[1])])
            bg = float(np.median(colmax))
            thr = bg + margin
            hit = set()
            for x in range(b["x0"] + 2, b["x1"] - 1):
                rr = hook_counter._longest_run_range(g[b["y0"]:b["y1"], x], thr)
                if len(rr) >= min_run and rr[0] <= min_top and rr[-1] >= min_bottom:
                    hit.add(x)
            if hit:
                cols_by_item[(i, os.path.basename(fp))] = hit
    if not cols_by_item:
        return []
    co = {}
    for hit in cols_by_item.values():
        cs = sorted(hit)
        for a in cs:
            for b in cs:
                d = b - a
                if min_gap <= d <= max_gap:
                    co[(a, b)] = co.get((a, b), 0) + 1
    if not co:
        return []
    best = max(co.items(), key=lambda kv: kv[1])
    a, b = best[0]
    return sorted([a, b])
