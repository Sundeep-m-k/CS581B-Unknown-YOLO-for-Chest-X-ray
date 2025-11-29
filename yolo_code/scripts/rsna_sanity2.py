import os
LBL_DIR="/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo/labels"
empty=nonempty=0
for f in os.listdir(LBL_DIR):
    if not f.endswith(".txt"): continue
    p=os.path.join(LBL_DIR,f)
    if os.path.getsize(p)==0: empty+=1
    else: nonempty+=1
print("No-box images:", empty, " With boxes:", nonempty)