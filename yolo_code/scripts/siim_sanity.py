import os

IMG_DIR="/home/sundeep/unknown_yolo_xray/datasets/siim_yolo_data/images"
LBL_DIR="/home/sundeep/unknown_yolo_xray/datasets/siim_yolo_data/labels"

imgs=[f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
lbls=[f for f in os.listdir(LBL_DIR) if f.endswith(".txt")]
print("Images:",len(imgs)," Labels:",len(lbls))

bad=[]
for lf in lbls:
    with open(os.path.join(LBL_DIR, lf)) as f:
        for ln in f:
            p=ln.strip().split()
            if not p or len(p)!=5:
                bad.append((lf, ln)); break
print("Bad label files:", len(bad), "Sample:", bad[:3])