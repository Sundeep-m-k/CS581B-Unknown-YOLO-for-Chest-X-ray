import os, random

IMG_DIR="/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo/images"
LBL_DIR="/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo/labels"

imgs=[f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
lbls=[f for f in os.listdir(LBL_DIR) if f.endswith(".txt")]
print("Images:",len(imgs)," Labels:",len(lbls))

# check label lines look like: class xc yc w h (5 tokens)
bad=[]
for lf in lbls:
    with open(os.path.join(LBL_DIR, lf)) as f:
        for ln in f:
            parts=ln.strip().split()
            if not(parts and (len(parts)==5)):
                bad.append((lf, ln))
                break
print("Bad label files:", len(bad))
print("Sample:", bad[:3])