import os, random, shutil
from pathlib import Path

BASE="/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo"
IMG=f"{BASE}/images"
LBL=f"{BASE}/labels"

files=[f for f in os.listdir(IMG) if f.endswith(".png")]
random.seed(42); random.shuffle(files)
cut=int(0.9*len(files)); train, val = files[:cut], files[cut:]

for split, subset in [("train",train),("val",val)]:
    Path(f"{IMG}/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"{LBL}/{split}").mkdir(parents=True, exist_ok=True)
    for fn in subset:
        stem=fn[:-4]
        shutil.move(f"{IMG}/{fn}", f"{IMG}/{split}/{fn}")
        if os.path.exists(f"{LBL}/{stem}.txt"):
            shutil.move(f"{LBL}/{stem}.txt", f"{LBL}/{split}/{stem}.txt")

print("Split done.")