import os, pydicom, cv2, pandas as pd
from pathlib import Path
from tqdm import tqdm
# ==== PATHS ====
IMG_DIR = "/home/sundeep/unknown_yolo_xray/datasets/rsna_pneumonia/stage_2_train_images"
CSV_PATH = "/home/sundeep/unknown_yolo_xray/datasets/rsna_pneumonia/stage_2_train_labels.csv"
OUT_IMG = "/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo/images"
OUT_LABEL = "/home/sundeep/unknown_yolo_xray/datasets/rsna_yolo/labels"

Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
Path(OUT_LABEL).mkdir(parents=True, exist_ok=True)
# ==== LOAD LABELS ====
df = pd.read_csv(CSV_PATH)

# ==== LOOP THROUGH IMAGES ====
for i, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['patientId']
    dicom_path = f"{IMG_DIR}/{img_id}.dcm"
    out_img_path = f"{OUT_IMG}/{img_id}.png"
    out_label_path = f"{OUT_LABEL}/{img_id}.txt"

    # Skip if file missing
    if not os.path.exists(dicom_path):
        continue

    # Read DICOM and save as PNG
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    cv2.imwrite(out_img_path, img)

    # If Target == 1 → pneumonia box exists
    if row['Target'] == 1:
        h, w = img.shape
        x, y, bw, bh = row['x'], row['y'], row['width'], row['height']
        # YOLO normalized center format
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h
        with open(out_label_path, "a") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        # No pneumonia → empty label file
        open(out_label_path, "w").close()

print("✅ Conversion complete! Images and labels saved in rsna_yolo/")