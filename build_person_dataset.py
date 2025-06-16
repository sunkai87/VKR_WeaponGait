# from pathlib import Path
# import csv

# root = Path("cache/poses_mp")
# out  = Path("data/manifest_person.csv")

# with out.open("w", newline="") as fh:
#     wr = csv.writer(fh)
#     wr.writerow(["video", "label"])               # header


#     # for npy in root.glob("*.npy"):
#     #     stem = npy.stem.split("_pid")[0]
#     #     label = "weapon" if "w_" in stem else "no_weapon"   # <-- простое правило
#     #     wr.writerow([npy.as_posix(), label])
    



#     MAX_PER_VIDEO = 2
#     counts = {}
#     for npy in root.glob('*.npy'):
#         vid = npy.stem.split('_pid')[0]
#         if counts.get(vid, 0) >= MAX_PER_VIDEO:
#             continue
#         counts[vid] = counts.get(vid, 0) + 1
#         label = 'weapon' if 'w_' in vid else 'no_weapon'
#         wr.writerow([npy, label])







# My working way
# root = Path("cache/poses_crop_mp")
# with open("data/manifest_crop.csv", "w", newline="") as f:
#     wr = csv.writer(f); wr.writerow(["video", "label"])
#     for npy in root.glob("*.npy"):
#         label = "weapon" if "w_" in npy.stem else "no_weapon"
#         wr.writerow([npy.as_posix(), label])

from pathlib import Path
import csv

root = Path("cache/poses_crop_mp")
with open("data/manifest_crop.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["video", "label"])
    MAX_PER_VIDEO = 8
    counts = {}
    for npy in root.glob('*.npy'):
        vid = npy.stem.split('_pid')[0]
        if counts.get(vid, 0) >= MAX_PER_VIDEO:
            continue
        counts[vid] = counts.get(vid, 0) + 1
        label = 'weapon' if 'w_' in vid else 'no_weapon'
        wr.writerow([npy.as_posix(), label])
    # for npy in root.glob("*.npy"):
    #     label = "weapon" if "w_" in npy.stem else "no_weapon"
    #     wr.writerow([npy.as_posix(), label])