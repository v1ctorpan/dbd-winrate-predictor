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
REPORT_DIR = os.path.join(BASE, "report")

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


def classify(crop, slot, refs):
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
    healthy_refs = refs.get("healthy", [])
    if 0 <= slot < len(healthy_refs):
        s = state_recognizer.ncc(crop, fit(healthy_refs[slot]))
        if s < 0.35:
            return "unknown"
        cur = state_recognizer.redness(crop)
        base = state_recognizer.redness(healthy_refs[slot])
        if cur["rg"] > base["rg"] + 12.0 and cur["sat"] > base["sat"] + 15.0:
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


def process_video(name, frame_dir, get_anchor, sample=None, seed=42):
    frame_names = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if sample is not None and sample < len(frame_names):
        import random
        rng = random.Random(seed)
        frame_names = sorted(rng.sample(frame_names, sample))
    cfg = hud_regions.load_regions(CFG)
    tpl = cv2.imread(ANCHOR_TPL)
    digit_refs = gens_counter.build_digit_refs()
    gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))

    out_dir = os.path.join(REPORT_DIR, name)
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    # 用第一帧确定锚点 + 槽位
    frame0 = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    anchor = get_anchor(frame0, tpl)
    if anchor is None:
        print(f"[{name}] 第一帧无锚点，跳过")
        return
    scale = anchor["scale"]
    resolved = hud_regions.resolve_regions(cfg, anchor)
    apply_hook_cfg(resolved, anchor, name)
    refs = build_refs(scale)
    slot_files = [os.path.join(frame_dir, f) for f in frame_names]
    slots = calibrator.calibrate_hook_slots(slot_files, resolved)
    print(f"[{name}] anchor=({anchor['x']},{anchor['y']}) scale={scale:.2f} slots={slots}")

    header = ["frame", "scale", "p1", "p2", "p3", "p4",
              "hooks", "gens", "机器标注"]
    rows = []
    for fname in frame_names:
        frame = cv2.imread(os.path.join(frame_dir, fname))
        anchor = get_anchor(frame, tpl) if get_anchor else anchor
        if anchor is None:
            rows.append([fname, "-", "-", "-", "-", "-", "-", "-", "无 HUD"])
            continue
        resolved = hud_regions.resolve_regions(cfg, anchor)
        apply_hook_cfg(resolved, anchor, name)
        states = []
        for i in range(1, 5):
            b = resolved[f"survivor_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            states.append(classify(crop, i - 1, refs))
        hooks = hook_counter.count_all(frame, resolved, slots)
        gens = gens_counter.count_gens(frame, resolved, digit_refs, gen=gen, anchor=anchor)
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
