import glob
import cv2
from ultralytics import YOLO
import numpy as np
import torch

DIR = "../dataset/images"
SAVE_DIR = "test"
imgs = glob.glob(f'{DIR}/*.png')
model = YOLO("best.pt")

for img_file in imgs:
    img = cv2.imread(img_file)
    result = model(img)[0]

    map = np.zeros(img.shape, dtype=np.uint8)
    
    if result.masks != None:
        boxes = result.boxes  # 경계 상자 정보
        masks = result.masks  # 마스크 정보

        for box, mask in zip(boxes, masks):
            cls_index = int(box.cls) 
            if cls_index == 0:
                color = (255,0,0)
            else:
                color = (0, 255, 0)
            cv2.drawContours(map, [np.array(mask.xy).astype(np.int32)], 0, color, -1)
            
    cv2.addWeighted(img, 0.5, map, 0.5, 0, map)

    gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    save_img = np.hstack(([map, img]))
    cv2.resize(save_img, (save_img.shape[1]//3, save_img.shape[0]//3), interpolation=cv2.INTER_LINEAR)

    #get filename from os.path
    filename = img_file.replace("\\", "/").split('/')[-1]
    save_path = f"{SAVE_DIR}/{filename}"
    cv2.imwrite(save_path, save_img)