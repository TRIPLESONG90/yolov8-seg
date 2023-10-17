from multiprocessing import freeze_support
from ultralytics import YOLO

from labelme2yolov8 import convert_label
import os
import time
def train():
    SRC_DIR = "/mnt/volume1/strap-segmentation"
    DST_DIR = "dataset"
    LABELS = ["OK"]

    os.path.isdir(DST_DIR) and os.system(f"rm -rf {DST_DIR}")

    convert_label(SRC_DIR, DST_DIR, LABELS)

    model = YOLO("./yolov8n-seg.pt", "segmentation")

    #get full path of DST_DIR
    DST_DIR = os.path.abspath(DST_DIR)

    #remove old train.yaml
    os.path.isfile("train.yaml") and os.system("rm train.yaml")

    #save train.yaml
    with open("train.yaml", 'w') as f:
        f.write(f"path: {DST_DIR}\n")
        f.write(f"train: test.txt\n")
        f.write(f"val: test.txt\n")
        f.write(f"test: test.txt\n")
        f.write(f"names: \n")
        for i in range(len(LABELS)):
            f.write(f"  {i}: {LABELS[i]}\n")

    model.train(data="./train.yaml", epochs=800, workers=1, batch=40, imgsz=640, patience=0,
                hsv_h = 0,
                hsv_s = 0,
                hsv_v = 0.05,
                )

if __name__ == "__main__":
    freeze_support()
    train()