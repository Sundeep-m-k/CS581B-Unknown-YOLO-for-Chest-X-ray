import os

BASE = "/home/sundeep/unknown_yolo_xray/datasets/siim_yolo_data"
paths = [
    f"{BASE}/labels",
    f"{BASE}/labels/train",
    f"{BASE}/labels/val",
]

for p in paths:
    n = sum(1 for f in os.listdir(p) if f.endswith(".txt")) if os.path.exists(p) else 0
    print(p, "exists:", os.path.exists(p), " txt files:", n)

# Count boxes (non-empty) recursively
nonempty = empty = 0
for root, _, files in os.walk(f"{BASE}/labels"):
    for f in files:
        if not f.endswith(".txt"): continue
        fp = os.path.join(root, f)
        if os.path.getsize(fp) == 0: empty += 1
        else: nonempty += 1
print("Recursive -> No-box images:", empty, " With boxes:", nonempty)