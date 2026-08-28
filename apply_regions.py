import os
import sys

import cv2

import hud_anchor
import hud_regions

CFG_PATH = r"D:\files\code\dbd_pred\config\hud_regions.json"
SRC_DIR = r"D:\files\code\dbd_pred\picture\test1"
OUT_DIR = r"D:\files\code\dbd_pred\picture\test1_regions"

def main():
    config = hud_regions.load_regions(CFG_PATH)
    template = cv2.imread(config["template"])
    os.makedirs(OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(SRC_DIR, fname))
        anchor = config["anchor"]
        resolved = hud_regions.resolve_regions(config, anchor)
        for name, box in resolved.items():
            cv2.rectangle(frame, (box["x0"], box["y0"]), (box["x1"], box["y1"]), (0, 255, 0), 2)
            cv2.putText(frame, name, (box["x0"] + 2, max(12, box["y0"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        a = anchor
        cv2.rectangle(frame, (a["x"], a["y"]), (a["x"] + a["w"], a["y"] + a["h"]), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, fname), frame)
        print(f"{fname}: done")

if __name__ == "__main__":
    main()
