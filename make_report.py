import os

import cv2
import numpy as np

import calibrator
import gens_counter
import hook_counter
import hud_anchor
import hud_regions
import state_recognizer

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(BASE, "config", "hud_regions.json")
HOOK_CFG = os.path.join(BASE, "config", "hook_regions.json")
ANCHOR_TPL = os.path.join(BASE, "picture", "gen.jpg")
CROPS = os.path.join(BASE, "picture", "crops")
ASSETS = os.path.join(BASE, "asset")
REPORT_DIR = os.path.join(BASE, "report")

OFFICIAL_ICONS = {
    "hooked": "icon_hook.webp",
    "dying": "icon_dying.webp",
    "escaped": "icon_exitGate.webp",
    "dead": "icon_scarified.png",
}


def load_official_icons():
    tpl = {}
    for state, fname in OFFICIAL_ICONS.items():
        img = cv2.imread(os.path.join(ASSETS, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            a = img[:, :, 3]
            ys, xs = np.nonzero(a > 0)
            bgr = img[:, :, :3].copy()
            bgr[a == 0] = 0
            tpl[state] = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        else:
            tpl[state] = img
    return tpl

REF_CROPS = {
    "hooked":  [("frame_0003", "survivor_p3")],
    "dying":   [("frame_0003", "survivor_p4")],
    "dead":    [("frame_0009", "survivor_p4")],
    "escaped": [("frame_0011", "survivor_p3")],
    "healthy": [("frame_0001", "survivor_p1"),
                ("frame_0001", "survivor_p2"),
                ("frame_0001", "survivor_p3"),
                ("frame_0001", "survivor_p4")],
    "injured": [("frame_0002", "survivor_p4")],
}

STATE_COLOR = {
    "healthy": (0, 255, 0),
    "injured": (0, 165, 255),
    "hooked":  (255, 0, 0),
    "dying":   (0, 0, 255),
    "dead":    (128, 128, 128),
    "escaped": (255, 255, 0),
    "unknown": (0, 0, 0),
}
BOX_COLOR = (255, 200, 0)
GENS_COLOR = (255, 0, 255)


def build_refs(scale):
    refs = {}
    for state, pairs in REF_CROPS.items():
        imgs = []
        for f, r in pairs:
            img = cv2.imread(os.path.join(CROPS, f, r + ".jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            scaled = cv2.resize(img, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
            imgs.append(scaled)
        if imgs:
            refs[state] = imgs
    return refs


def classify(crop, slot, refs, icon_tpl=None):
    h, w = crop.shape[:2]

    def fit(img):
        if img.shape[:2] != (h, w):
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    best_icon, best_score = None, -1.0
    for state in state_recognizer.ICON_STATES:
        for ref in refs.get(state, []):
            s = state_recognizer.ncc(crop, fit(ref))
            if s > best_score:
                best_score, best_icon = s, state
    if best_score >= 0.55:
        return best_icon
    if icon_tpl:
        for state, tpl in icon_tpl.items():
            s = state_recognizer.multi_scale_match(crop, tpl)
            if s > best_score:
                best_score, best_icon = s, state
        if best_score >= 0.70:
            return best_icon
    healthy_refs = refs.get("healthy", [])
    if 0 <= slot < len(healthy_refs):
        s = state_recognizer.ncc(crop, fit(healthy_refs[slot]))
        if s < 0.35:
            return "unknown"
        if state_recognizer.red_diag_clusters(crop, healthy_refs[slot]) >= 1:
            return "injured"
        return "healthy"
    return "unknown"


def draw_label(img, text, pos, color, scale=1.0, font_scale=0.5):
    x, y = int(pos[0]), int(pos[1])
    fs = font_scale * scale
    thickness = max(1, int(1 * scale))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
    y0 = y - 4 if y - 4 > th + 4 else y + th + 8
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0 + 4), color, -1)
    cv2.putText(img, text, (x + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255) if np.mean(color) < 128 else (0, 0, 0), thickness)


def annotate_frame(frame, resolved, slots, states, hooks, gens, scale):
    out = frame.copy()
    s = max(scale, 1.0)
    for i in range(1, 5):
        b = resolved[f"survivor_p{i}"]
        color = STATE_COLOR.get(states[i - 1], (0, 0, 0))
        cv2.rectangle(out, (b["x0"], b["y0"]), (b["x1"], b["y1"]), color, max(2, int(2 * scale)))
        draw_label(out, f"p{i} {states[i - 1]}", (b["x0"], b["y0"]), color, s)
    for i in range(1, 5):
        b = resolved[f"hook_p{i}"]
        cv2.rectangle(out, (b["x0"], b["y0"]), (b["x1"], b["y1"]), BOX_COLOR, max(1, int(1 * scale)))
        draw_label(out, f"h{i}={hooks[i - 1]}", (b["x1"] + 4, b["y0"] + (b["y1"] - b["y0"]) // 2), BOX_COLOR, s)
        for slot in slots:
            cv2.line(out, (slot, b["y0"]), (slot, b["y0"] + 2), (255, 255, 255), 1)
    b = resolved["gens_row"]
    cv2.rectangle(out, (b["x0"], b["y0"]), (b["x1"], b["y1"]), GENS_COLOR, max(1, int(1 * scale)))
    draw_label(out, f"gens={gens}", (b["x0"], b["y0"]), GENS_COLOR, s)
    return out


def apply_hook_cfg(resolved, anchor, video_name=None):
    if not os.path.exists(HOOK_CFG):
        return
    with open(HOOK_CFG, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
    if video_name is not None and data.get("video") != video_name:
        return
    for name, rel in data["regions"].items():
        if name in resolved:
            resolved[name] = hud_regions.rel_to_abs(rel, anchor)


def build_opening_refs(frame, resolved):
    healthy = []
    for i in range(1, 5):
        b = resolved[f"survivor_p{i}"]
        healthy.append(frame[b["y0"]:b["y1"], b["x0"]:b["x1"]])
    return {"healthy": healthy}


def pick_opening_frame(frame_dir, names, get_anchor, tpl, cfg, max_probe=8):
    anchor0 = None
    for k in range(min(max_probe, len(names))):
        frame = cv2.imread(os.path.join(frame_dir, names[k]))
        if frame is None:
            continue
        anchor = get_anchor(frame, tpl)
        if anchor is None:
            continue
        resolved = hud_regions.resolve_regions(cfg, anchor)
        crops = [frame[r["y0"]:r["y1"], r["x0"]:r["x1"]]
                 for r in [resolved[f"survivor_p{i}"] for i in range(1, 5)]]
        total = 0
        nxt = k + 1
        if nxt >= len(names):
            return frame, anchor, resolved
        frame_next = cv2.imread(os.path.join(frame_dir, names[nxt]))
        if frame_next is None:
            return frame, anchor, resolved
        anchor_next = get_anchor(frame_next, tpl) or anchor
        resolved_next = hud_regions.resolve_regions(cfg, anchor_next)
        for i in range(4):
            b = resolved_next[f"survivor_p{i + 1}"]
            cur = frame_next[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            h, w = cur.shape[:2]
            total += state_recognizer.ncc(crops[i], cv2.resize(cur, (w, h)))
        if total / 4.0 >= 0.7:
            return frame, anchor, resolved
        anchor0 = anchor
    if anchor0 is None:
        return None, None, None
    frame = cv2.imread(os.path.join(frame_dir, names[0]))
    return frame, anchor0, hud_regions.resolve_regions(cfg, anchor0)


def _detect_match_swaps(frame_dir, all_names, cfg, tpl, gen, anchor,
                        resolved, digit_refs, get_anchor, step=10):
    """稀疏采样 gens，返回换局点帧名列表（gens 从非 5 跳回 5 的位置）。
    换局后幸存者角色可能变化，需为每局重建 healthy 参考。"""
    swaps = []
    prev_g = None
    for i in range(0, len(all_names), step):
        frame = cv2.imread(os.path.join(frame_dir, all_names[i]))
        if frame is None:
            continue
        cur = get_anchor(frame, tpl) or anchor
        r = hud_regions.resolve_regions(cfg, cur)
        apply_hook_cfg(r, cur, "BV1Uu8z6eEVM")
        g = gens_counter.count_gens(frame, r, digit_refs, gen=gen, anchor=cur)
        if prev_g is not None and prev_g not in (5, None) and g == 5:
            # 精化：从采样点向前找第一个 gens=5 的帧作为精确换局帧
            exact = _find_swap_start(frame_dir, all_names, i, cfg, tpl, gen,
                                     anchor, digit_refs, get_anchor)
            swaps.append(exact)
        prev_g = g
    return swaps


def _find_swap_start(frame_dir, all_names, idx, cfg, tpl, gen, anchor,
                     digit_refs, get_anchor, back=30):
    """从采样点 idx 向前找换局精确起点：跳过前导 gens=5 段，
    找到最后一个 gens!=5 的帧，其后的第一个 gens=5 帧即换局起点。"""
    start = max(idx - back, 0)
    last_non5 = None
    for j in range(idx, start - 1, -1):
        frame = cv2.imread(os.path.join(frame_dir, all_names[j]))
        if frame is None:
            continue
        cur = get_anchor(frame, tpl) or anchor
        r = hud_regions.resolve_regions(cfg, cur)
        apply_hook_cfg(r, cur, "BV1Uu8z6eEVM")
        g = gens_counter.count_gens(frame, r, digit_refs, gen=gen, anchor=cur)
        if g != 5:
            last_non5 = j
            break
    if last_non5 is None:
        return all_names[idx]
    for j in range(last_non5 + 1, last_non5 + 1 + back):
        if j >= len(all_names):
            break
        frame = cv2.imread(os.path.join(frame_dir, all_names[j]))
        if frame is None:
            continue
        cur = get_anchor(frame, tpl) or anchor
        r = hud_regions.resolve_regions(cfg, cur)
        apply_hook_cfg(r, cur, "BV1Uu8z6eEVM")
        g = gens_counter.count_gens(frame, r, digit_refs, gen=gen, anchor=cur)
        if g == 5:
            return all_names[j]
    return all_names[idx]


def process_video(name, frame_dir, get_anchor, sample=None, seed=42):
    all_names = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    cfg = hud_regions.load_regions(CFG)
    tpl = cv2.imread(ANCHOR_TPL)
    gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))

    out_dir = os.path.join(REPORT_DIR, name)
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    # 自动挑选开局已渲染 HUD 的帧，以其各角色形象作为第 1 局 healthy 参考
    opening, anchor, resolved = pick_opening_frame(frame_dir, all_names, get_anchor, tpl, cfg)
    if anchor is None:
        print(f"[{name}] 开局帧无锚点，跳过")
        return
    global_anchor = anchor
    scale = anchor["scale"]
    apply_hook_cfg(resolved, anchor, name)
    refs = build_refs(scale)
    refs["healthy"] = build_opening_refs(opening, resolved)["healthy"]
    icon_tpl = load_official_icons()
    digit_refs = gens_counter.load_digit_refs()
    gens_tracker = gens_counter.GensTracker(digit_refs, gen=gen)

    # 稀疏采样检测换局点：gens 从非 5 跳回 5 视为新一局开始
    swap_frames = _detect_match_swaps(
        frame_dir, all_names, cfg, tpl, gen, anchor, resolved,
        digit_refs, get_anchor)

    frame_names = all_names
    if sample is not None and sample < len(frame_names):
        import random
        rng = random.Random(seed)
        frame_names = sorted(rng.sample(frame_names, sample))
    slot_files = [os.path.join(frame_dir, f) for f in frame_names]
    slots = calibrator.calibrate_hook_slots(slot_files, resolved)
    print(f"[{name}] anchor=({anchor['x']},{anchor['y']}) scale={scale:.2f} slots={slots}")

    header = ["frame", "scale", "p1", "p2", "p3", "p4",
              "hooks", "gens", "机器标注"]
    rows = []
    swap_set = set(swap_frames)
    for fname in frame_names:
        frame = cv2.imread(os.path.join(frame_dir, fname))
        cur_anchor = get_anchor(frame, tpl) if get_anchor else None
        anchor = cur_anchor if cur_anchor is not None else global_anchor
        resolved = hud_regions.resolve_regions(cfg, anchor)
        apply_hook_cfg(resolved, anchor, name)
        if fname in swap_set:
            refs["healthy"] = build_opening_refs(frame, resolved)["healthy"]
            gens_tracker.reset()
        states = []
        for i in range(1, 5):
            b = resolved[f"survivor_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            states.append(classify(crop, i - 1, refs, icon_tpl))
        hooks = hook_counter.count_all(frame, resolved, slots)
        gens = gens_tracker.update(frame, resolved, anchor)
        flags = []
        for i, st in enumerate(states):
            if st == "unknown":
                flags.append(f"p{i + 1}未知")
        if gens is None:
            flags.append("gens未识别")
        note = "; ".join(flags) if flags else "正常"
        rows.append([
            fname, f"{scale:.2f}",
            states[0], states[1], states[2], states[3],
            "/".join(str(h) for h in hooks),
            str(gens), note,
        ])
        out = annotate_frame(frame, resolved, slots, states, hooks, gens, scale)
        cv2.imwrite(os.path.join(ann_dir, fname), out)

    with open(os.path.join(out_dir, "detect_report.csv"), "w", newline="", encoding="utf-8-sig") as f:
        import csv
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[{name}] {len(rows)} 帧 -> {out_dir}")


def main():
    cfg = hud_regions.load_regions(CFG)
    process_video(
        "test1",
        os.path.join(BASE, "picture", "test1"),
        lambda frame, tpl: cfg["anchor"],
    )
    process_video(
        "BV1Uu8z6eEVM",
        os.path.join(BASE, "picture", "BV1Uu8z6eEVM"),
        lambda frame, tpl: hud_anchor.detect_anchor(frame, tpl, prior=(121, 847)),
        sample=10,
    )


if __name__ == "__main__":
    main()
