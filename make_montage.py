import os

import cv2
import numpy as np

CROPS_DIR = r"D:\files\code\dbd_pred\picture\crops"
OUT_PATH = r"D:\files\code\dbd_pred\picture\portrait_montage.png"

SCALE = 3
SLOTS = ["survivor_p1", "survivor_p2", "survivor_p3", "survivor_p4"]

def main():
    frames = sorted(os.listdir(CROPS_DIR))
    rows = []
    for f in frames:
        crops = []
        for s in SLOTS:
            p = os.path.join(CROPS_DIR, f, f"{s}.jpg")
            img = cv2.imread(p)
            if img is not None:
                img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            else:
                img = np.zeros((44 * SCALE, 48 * SCALE, 3), dtype=np.uint8)
            crops.append(img)
        row = np.hstack(crops)
        canvas = np.full((row.shape[0] + 40, row.shape[1], 3), 30, dtype=np.uint8)
        canvas[:row.shape[0]] = row
        cv2.putText(canvas, f, (10, row.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(canvas)
    montage = np.vstack(rows)
    cv2.imwrite(OUT_PATH, montage)
    print("saved", OUT_PATH, montage.shape)

if __name__ == "__main__":
    main()
