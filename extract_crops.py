import os

import cv2

import hud_regions

CFG_PATH = r"D:\files\code\dbd_pred\config\hud_regions.json"
SRC_DIR = r"D:\files\code\dbd_pred\picture\test1"
OUT_DIR = r"D:\files\code\dbd_pred\picture\crops"

REGIONS_OF_INTEREST = ["survivor_p1", "survivor_p2", "survivor_p3", "survivor_p4"]

def main():
    config = hud_regions.load_regions(CFG_PATH)
    anchor = config["anchor"]
    resolved = hud_regions.resolve_regions(config, anchor)
    os.makedirs(OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(SRC_DIR, fname))
        stem = os.path.splitext(fname)[0]
        for name in REGIONS_OF_INTEREST:
            b = resolved[name]
            crop = frame[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            sub = os.path.join(OUT_DIR, stem)
            os.makedirs(sub, exist_ok=True)
            cv2.imwrite(os.path.join(sub, f"{name}.jpg"), crop)
    print("crops saved to", OUT_DIR)

if __name__ == "__main__":
    main()
