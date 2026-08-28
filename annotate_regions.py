import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = r"D:\files\code\dbd_pred\picture\test1"
OUT_DIR = r"D:\files\code\dbd_pred\picture\test1_annotated"
FRAMES = ["frame_0000.jpg", "frame_0006.jpg", "frame_0010.jpg"]

W, H = 1280, 720

REGIONS = [
    ("survivor_p1",       0.012, 0.480, 0.078, 0.600, (0, 255, 0)),
    ("survivor_p2",       0.012, 0.604, 0.078, 0.724, (0, 255, 0)),
    ("survivor_p3",       0.012, 0.728, 0.078, 0.848, (0, 255, 0)),
    ("survivor_p4",       0.012, 0.852, 0.078, 0.972, (0, 255, 0)),
    ("hook_pips",         0.082, 0.480, 0.130, 0.972, (255, 200, 0)),
    ("gens_row",          0.320, 0.120, 0.660, 0.185, (0, 200, 255)),
    ("gates_icons",       0.665, 0.120, 0.780, 0.185, (255, 0, 255)),
    ("timer",             0.240, 0.120, 0.318, 0.185, (255, 255, 0)),
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for fname in FRAMES:
        img = cv2.imread(os.path.join(SRC_DIR, fname))
        if img is None:
            print(f"skip {fname}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for name, x0, y0, x1, y1, color in REGIONS:
            x0p, y0p = int(x0 * W), int(y0 * H)
            x1p, y1p = int(x1 * W), int(y1 * H)
            draw.rectangle([x0p, y0p, x1p, y1p], outline=color, width=3)
            draw.text((x0p + 3, y0p - 22), name, fill=color, font=font)
        out_path = os.path.join(OUT_DIR, fname)
        pil.save(out_path)
        print(f"saved {out_path}")

if __name__ == "__main__":
    main()
