import os

import cv2
import numpy as np

CROPS_DIR = r"D:\files\code\dbd_pred\picture\crops"

ICON_STATES = ["hooked", "dying", "dead", "escaped"]
FACE_STATES = ["healthy", "injured"]

EXAMPLE_REFS = {
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

def load_crop(frame_stem, region):
    return cv2.imread(os.path.join(CROPS_DIR, frame_stem, f"{region}.jpg"))

def build_references():
    refs = {}
    for state, pairs in EXAMPLE_REFS.items():
        imgs = [load_crop(f, r) for f, r in pairs]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            refs[state] = imgs
    return refs

def ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)

def redness(crop):
    b = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "rg": float((r - g).mean()),
        "sat": float(hsv[:, :, 1].mean()),
    }

def classify(crop, slot, refs, icon_thr=0.55, face_thr=0.35, inj_rg_delta=12.0, inj_sat_delta=15.0):
    best_icon, best_score = None, -1.0
    for state in ICON_STATES:
        for ref in refs.get(state, []):
            s = ncc(crop, ref)
            if s > best_score:
                best_score, best_icon = s, state
    if best_score >= icon_thr:
        return best_icon, best_score

    healthy_refs = refs.get("healthy", [])
    if 0 <= slot < len(healthy_refs):
        s = ncc(crop, healthy_refs[slot])
        if s < face_thr:
            return "unknown", s
        cur = redness(crop)
        base = redness(healthy_refs[slot])
        if cur["rg"] > base["rg"] + inj_rg_delta and cur["sat"] > base["sat"] + inj_sat_delta:
            return "injured", s
        return "healthy", s
    return "unknown", best_score

def main():
    refs = build_references()
    for state, imgs in refs.items():
        print(f"{state}: {[i.shape for i in imgs]}")
    print()
    print(f"{'frame':14} {'p1':14} {'p2':14} {'p3':14} {'p4':14}")
    for f in sorted(os.listdir(CROPS_DIR)):
        row = []
        for i in range(1, 5):
            crop = load_crop(f, f"survivor_p{i}")
            if crop is None:
                row.append("----")
                continue
            state, score = classify(crop, i - 1, refs)
            row.append(f"{state}({score:.2f})")
        print(f"{f:14} {row[0]:14} {row[1]:14} {row[2]:14} {row[3]:14}")

if __name__ == "__main__":
    main()
