import os

import cv2
import numpy as np

TEMPLATE_PATH = r"D:\files\code\dbd_pred\picture\gen.jpg"
SRC_DIR = r"D:\files\code\dbd_pred\picture\test1"
OUT_DIR = r"D:\files\code\dbd_pred\picture\test1_genmatch"

SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

def find_icon(frame, template, scales):
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    best = None
    for s in scales:
        t = cv2.resize(template, (int(tw * s), int(th * s)), interpolation=cv2.INTER_AREA)
        if t.shape[0] >= fh or t.shape[1] >= fw:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if best is None or maxv > best[0]:
            best = (maxv, s, maxloc, t.shape[:2])
    return best

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
        score, s, (x, y), (th, tw) = find_icon(frame, template, SCALES)
        out = frame.copy()
        cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 0, 255), 3)
        cv2.putText(out, f"{score:.3f} scale={s:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, out)
        print(f"{fname}: best_score={score:.3f} scale={s:.2f} pos=({x},{y}) size=({tw}x{th})")

if __name__ == "__main__":
    main()
