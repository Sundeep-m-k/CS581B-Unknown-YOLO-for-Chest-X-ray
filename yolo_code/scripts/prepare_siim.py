# SIIM Pneumothorax RLE → YOLO format (Simple version)
import os, cv2, pydicom, numpy as np, pandas as pd
from tqdm import tqdm
from pathlib import Path

# ==== PATHS ====
IMG_DIR = "/home/sundeep/unknown_yolo_xray/datasets/siim_acr_correct/stage_2_train"
CSV_PATH = "/home/sundeep/unknown_yolo_xray/datasets/siim_acr_correct/stage_2_train.csv"
OUT_IMG = "/home/sundeep/unknown_yolo_xray/datasets/siim_yolo_data/images"
OUT_LABEL = "/home/sundeep/unknown_yolo_xray/datasets/siim_yolo_data/labels"

Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
Path(OUT_LABEL).mkdir(parents=True, exist_ok=True)

# ==== LOAD CSV ====
df = pd.read_csv(CSV_PATH)
print("Total rows:", len(df))

# ==== RLE DECODER ====
def rle_decode(mask_rle, shape=(1024, 1024)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # flip axes

# ==== LOOP THROUGH IMAGES ====
for img_id, group in tqdm(df.groupby('ImageId')):
    dicom_path = f"{IMG_DIR}/{img_id}.dcm"
    if not os.path.exists(dicom_path):
        continue

    # Read image
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    h, w = img.shape

    # Save PNG
    out_img_path = f"{OUT_IMG}/{img_id}.png"
    cv2.imwrite(out_img_path, img)

    # Label path
    out_label_path = f"{OUT_LABEL}/{img_id}.txt"

    # Check if this image has pneumothorax mask(s)
    if group['EncodedPixels'].isnull().all():
        open(out_label_path, "w").close()
        continue

    # Decode each mask → find bounding box
    with open(out_label_path, "w") as f:
        for mask_rle in group['EncodedPixels'].dropna():
            mask = rle_decode(mask_rle, (h, w))
            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ SIIM RLE → YOLO conversion complete! Saved in siim_yolo_data/")