import os

import cv2
import numpy as np

import hud_regions

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "picture", "test1")
CFG = os.path.join(BASE, "config", "hud_regions.json")
GEN_TPL = os.path.join(BASE, "picture", "gen.jpg")

DIGIT_X = 29
GEN_ICON_THR = 0.70
DIGIT_THR = 0.55

DIGIT_REFS_SRC = {
    5: ["frame_0001"],
    4: ["frame_0002", "frame_0003", "frame_0004", "frame_0005"],
    3: ["frame_0006"],
    2: ["frame_0007", "frame_0008", "frame_0009"],
}

def _ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)

def _match_gen_icon(crop, gen, anchor):
    s = anchor["scale"]
    th, tw = gen.shape[:2]
    w, h = int(tw * s), int(th * s)
    t = cv2.resize(gen, (w, h), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return float(maxv)

def build_digit_refs():
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    b = resolved["gens_row"]
    refs = {}
    for digit, frames in DIGIT_REFS_SRC.items():
        imgs = []
        for f in frames:
            frame = cv2.imread(os.path.join(SRC, f + ".jpg"))
            if frame is None:
                continue
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            imgs.append(crop[:, 0:DIGIT_X])
        if imgs:
            refs[digit] = imgs
    return refs

def count_gens(frame, resolved, refs, gen=None, anchor=None, gen_icon_thr=GEN_ICON_THR, digit_thr=DIGIT_THR):
    b = resolved["gens_row"]
    crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if int((g > 100).sum()) < 20:
        return None

    if gen is None:
        gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))
    if anchor is None:
        anchor = {"scale": 1.0}

    if _match_gen_icon(crop, gen, anchor) >= gen_icon_thr:
        digit_w = int(DIGIT_X * anchor["scale"])
        digit_crop = crop[:, 0:digit_w]
        dh, dw = digit_crop.shape[:2]

        def fit(img):
            if img.shape[:2] != (dh, dw):
                return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            return img

        best, best_score = None, -1.0
        for digit, imgs in refs.items():
            for ref in imgs:
                s = _ncc(digit_crop, fit(ref))
                if s > best_score:
                    best_score, best = s, digit
        if best is not None and best_score >= digit_thr:
            return best
        return None
    return 0

def main():
    cfg = hud_regions.load_regions(CFG)
    cfg["template"] = os.path.join(BASE, "picture", "gen.jpg")
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    refs = build_digit_refs()
    gen = cv2.imread(os.path.join(BASE, "picture", "gen.jpg"))

    truth = {
        "frame_0000": None, "frame_0001": 5, "frame_0002": 4, "frame_0003": 4,
        "frame_0004": 4, "frame_0005": 4, "frame_0006": 3, "frame_0007": 2,
        "frame_0008": 2, "frame_0009": 2, "frame_0010": 0, "frame_0011": 0,
    }
    print(f"{'frame':12} {'gens':6} {'truth':6} {'ok'}")
    all_ok = True
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".jpg"):
            continue
        stem = fname[:-4]
        frame = cv2.imread(os.path.join(SRC, fname))
        got = count_gens(frame, resolved, refs, gen=gen, anchor=cfg["anchor"])
        exp = truth.get(stem)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"{stem:12} {str(got):6} {str(exp):6} {'OK' if ok else 'FAIL'}")
    print("ALL OK" if all_ok else "SOME FAILED")

if __name__ == "__main__":
    main()
