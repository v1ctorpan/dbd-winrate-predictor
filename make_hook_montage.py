import os

import cv2
import numpy as np

import hook_counter
import hud_regions

SRC = r"D:\files\code\dbd_pred\picture\test1"
CFG = r"D:\files\code\dbd_pred\config\hud_regions.json"
OUT = r"D:\files\code\dbd_pred\picture\hook_montage.png"
SCALE = 8

def main():
    cfg = hud_regions.load_regions(CFG)
    resolved = hud_regions.resolve_regions(cfg, cfg["anchor"])
    frames = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    cell_w = 23 * SCALE
    cell_h = 20 * SCALE
    pad = 10
    label_h = 24
    mont = np.zeros((label_h + 12 * (cell_h + pad), 4 * (cell_w + pad) + pad, 3),
                    dtype=np.uint8) + 20
    for r, fname in enumerate(frames):
        frame = cv2.imread(os.path.join(SRC, fname))
        y0 = label_h + r * (cell_h + pad)
        cv2.putText(mont, fname.replace(".jpg", ""), (pad, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for c in range(4):
            crop = hook_counter.hook_crop(frame, resolved, c + 1)
            up = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            x0 = pad + c * (cell_w + pad)
            mont[y0:y0 + cell_h, x0:x0 + cell_w] = up
            cv2.putText(mont, f"p{c + 1}", (x0, y0 + cell_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.imwrite(OUT, mont)
    print(f"saved {OUT} {mont.shape}")

if __name__ == "__main__":
    main()
