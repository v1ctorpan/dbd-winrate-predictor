import hud_regions

CFG_PATH = r"D:\files\code\dbd_pred\config\hud_regions.json"
CORRECT_ANCHOR = {"x": 94, "y": 536, "w": 35, "h": 32, "score": 0.9448, "scale": 1.0}

def main():
    config = hud_regions.load_regions(CFG_PATH)
    bad = config["anchor"]
    s_bad = bad["scale"]
    bx, by = bad["x"], bad["y"]

    for name in ("endgame_timer", "gate_ui", "hatch_ui"):
        if name not in config["regions"]:
            continue
        rel = config["regions"][name]
        abs_box = (
            bx + rel["x0"] * s_bad, by + rel["y0"] * s_bad,
            bx + rel["x1"] * s_bad, by + rel["y1"] * s_bad,
        )
        new_rel = hud_regions.abs_to_rel(abs_box, CORRECT_ANCHOR)
        config["regions"][name] = new_rel
        print(f"{name}: rel {rel} -> {new_rel}")

    config["anchor"] = CORRECT_ANCHOR
    hud_regions.save_regions(CFG_PATH, config)
    print("anchor restored, saved")

if __name__ == "__main__":
    main()
