import glob
import os
import json
import numpy as np
import cv2
import shutil
from tqdm import tqdm

def convert_label(SRC_DIR, DST_DIR, LABELS):
    os.path.isdir(DST_DIR) or os.makedirs(DST_DIR)

    
    j = json.load(open(f'{SRC_DIR}/annotations/instances_default.json'))

    categories = j["categories"]
    images = j["images"]

    dst_img_dir = os.path.join(DST_DIR, "images")
    os.path.isdir(dst_img_dir) or os.makedirs(dst_img_dir)
    dst_label_dir = os.path.join(DST_DIR, "labels")
    os.path.isdir(dst_label_dir) or os.makedirs(dst_label_dir)

    dst_imgs = []
    for i in tqdm(range(len(images)), desc="Converting labelme format to yolo format"):
        image = images[i]
        img_file = f'{SRC_DIR}/images/{image["file_name"]}'
        img = cv2.imread(img_file)
        h, w, c = img.shape
        annotations = list(filter(lambda x : x["image_id"] == image["id"], j["annotations"]))
        if len(annotations) == 0:
            continue
        dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_file))
        os.path.isfile(dst_img_path) or shutil.copy(img_file, dst_img_path)
        dst_imgs.append(dst_img_path)
        dst_txt_path = os.path.join(dst_label_dir, os.path.splitext(os.path.basename(img_file))[0] + ".txt")
        with open(dst_txt_path, 'w') as f:
            for annotation in annotations:
                # bbox = annotation["bbox"]
                index = annotation["category_id"] - 1

                pts = ""
                for s in annotation["segmentation"]:
                    seg = [[int(s[i]), int(s[i + 1])] for i in range(0, len(s), 2)]
                for s in seg:
                    x = s[0]
                    y = s[1]
                    pts += f'{x / w} {y / h} '

                line = f'{index} {pts}\n'
                f.write(line)

    with open(os.path.join(DST_DIR, "test.txt"), 'w') as f:
        for img in dst_imgs:
            f.write(img + "\n")

if __name__ == "__main__":
    SRC_DIR = "../dataset"
    DST_DIR = "dataset"
    LABELS = ["unoccluded", "occluded"]
    convert_label(SRC_DIR, DST_DIR, LABELS)