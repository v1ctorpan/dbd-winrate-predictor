import os

import cv2
import numpy as np

TEMPLATE_PATH = r"D:\files\code\dbd_pred\picture\gen.jpg"
SRC_DIR = r"D:\files\code\dbd_pred\picture\test1"
OUT_DIR = r"D:\files\code\dbd_pred\picture\test1_genmulti"

SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

def find_icons(frame, template, scales, top_n=6):
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    found = []
    for s in scales:
        t = cv2.resize(template, (int(tw * s), int(th * s)), interpolation=cv2.INTER_AREA)
        if t.shape[0] >= fh or t.shape[1] >= fw:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        while True:
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv < 0.60:
                break
            found.append((maxv, s, maxloc, t.shape[:2]))
            x, y = maxloc
            tth, ttw = t.shape[:2]
            res[y:y + tth, x:x + ttw] = -1.0
    found.sort(key=lambda r: -r[0])
    return found[:top_n]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print("template load failed")
        return

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".jpg"):
            continue
        path = os.path.join(SRC_DIR, fname)
        frame = cv2.imread(path)
        matches = find_icons(frame, template, SCALES)
        out = frame.copy()
        for i, (score, s, (x, y), (th, tw)) in enumerate(matches):
            cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 0, 255), 2)
            cv2.putText(out, f"{i}:{score:.2f}/{s:.2f}", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, out)
        desc = ", ".join(f"({x},{y})s={s:.2f}sc={score:.2f}" for score, s, (x, y), _ in matches)
        print(f"{fname}: {len(matches)} matches -> {desc}")

if __name__ == "__main__":
    main()
