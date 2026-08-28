import hud_regions

CFG_PATH = r"D:\files\code\dbd_pred\config\hud_regions.json"

ALIGNED_ABS = {
    "survivor_p1": (62, 282, 110, 326),
    "survivor_p2": (62, 341, 110, 385),
    "survivor_p3": (62, 400, 110, 444),
    "survivor_p4": (62, 459, 110, 503),
    "hook_p1":    (146, 283, 169, 303),
    "hook_p2":    (146, 342, 169, 362),
    "hook_p3":    (146, 401, 169, 421),
    "hook_p4":    (146, 460, 169, 480),
}

def main():
    config = hud_regions.load_regions(CFG_PATH)
    anchor = config["anchor"]
    for name, box in ALIGNED_ABS.items():
        config["regions"][name] = hud_regions.abs_to_rel(box, anchor)
        print(f"{name}: abs {box} -> rel {config['regions'][name]}")
    hud_regions.save_regions(CFG_PATH, config)
    print("saved")

if __name__ == "__main__":
    main()
