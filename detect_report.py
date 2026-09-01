import os
import random

import cv2
import numpy as np

import gens_counter
import hook_counter
import hud_anchor
import hud_regions
import state_recognizer

BASE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
CFG = os.path.join(BASE, "config", "hud_regions.json")
ANCHOR_TPL = os.path.join(BASE, "picture", "gen.jpg")

CROPS = os.path.join(BASE, "picture", "crops")
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


def load_ref(state, slot):
    pairs = REF_CROPS[state]
    return [cv2.imread(os.path.join(CROPS, f, r + ".jpg")) for f, r in pairs]


def build_refs(scale):
    refs = {}
    for state, pairs in REF_CROPS.items():
        imgs = []
        for f, r in pairs:
            img = cv2.imread(os.path.join(CROPS, f, r + ".jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            imgs.append(scaled)
        if imgs:
            refs[state] = imgs
    return refs


def classify(crop, slot, refs, icon_thr=0.55, face_thr=0.35, inj_rg_delta=12.0, inj_sat_delta=15.0):
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
    if best_score >= icon_thr:
        return best_icon, best_score
    healthy_refs = refs.get("healthy", [])
    if 0 <= slot < len(healthy_refs):
        s = state_recognizer.ncc(crop, fit(healthy_refs[slot]))
        if s < face_thr:
            return "unknown", s
        cur = state_recognizer.redness(crop)
        base = state_recognizer.redness(healthy_refs[slot])
        if cur["rg"] > base["rg"] + inj_rg_delta and cur["sat"] > base["sat"] + inj_sat_delta:
            return "injured", s
        return "healthy", s
    return "unknown", best_score


def main():
    import csv

    frames = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg"))
    random.seed(42)
    chosen = random.sample(frames, 10)

    cfg = hud_regions.load_regions(CFG)
    tpl = cv2.imread(ANCHOR_TPL)
    gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
    digit_refs = gens_counter.load_digit_refs()

    rows = []
    ref_anchor = (121, 847)  # 主流对局 HUD 锚点位置
    for fname in sorted(chosen):
        frame = cv2.imread(os.path.join(FRAME_DIR, fname))
        anchor = hud_anchor.detect_anchor(frame, tpl)
        if anchor is None:
            rows.append([fname, "-", "-", "-", "-", "-", "-", "无 HUD", ""])
            continue
        if abs(anchor["x"] - ref_anchor[0]) > 30 or abs(anchor["y"] - ref_anchor[1]) > 30:
            rows.append([fname, f"{anchor['scale']:.2f}",
                         "-", "-", "-", "-", "-",
                         f"锚点异常@({anchor['x']},{anchor['y']})", ""])
            continue
        scale = anchor["scale"]
        resolved = hud_regions.resolve_regions(cfg, anchor)
        refs = build_refs(scale)

        states = []
        for i in range(1, 5):
            b = resolved[f"survivor_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            st, sc = classify(crop, i - 1, refs)
            states.append(st)

        hooks = []
        for i in range(1, 5):
            b = resolved[f"hook_p{i}"]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            hooks.append(hook_counter.count_hooks(crop))

        gens = gens_counter.count_gens(frame, resolved, digit_refs, gen=gen, anchor=anchor)

        flags = []
        for i, s in enumerate(states):
            if s == "unknown":
                flags.append(f"p{i+1}未知")
        if gens is None:
            flags.append("gens未识别")
        note = "; ".join(flags) if flags else "正常"

        rows.append([
            fname,
            f"{scale:.2f}",
            states[0], states[1], states[2], states[3],
            "/".join(str(h) for h in hooks),
            str(gens),
            note,
            "",  # 人工备注
        ])

    header = ["frame", "scale", "p1", "p2", "p3", "p4", "hooks", "gens", "机器标注", "人工备注(请填写)"]

    csv_path = os.path.join(BASE, "detect_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV saved -> {csv_path}")

    md_path = os.path.join(BASE, "detect_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print(f"MD saved -> {md_path}")

    # 控制台简要输出
    for r in rows:
        print(f"{r[0]:20} scale={r[1]:>4} p1={r[2]:7} p2={r[3]:7} p3={r[4]:7} p4={r[5]:7} hooks={r[6]:5} gens={r[7]:4} [{r[8]}]")


if __name__ == "__main__":
    main()
