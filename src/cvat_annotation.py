import glob
import cv2
from ultralytics import YOLO
import numpy as np
import json
import os
IMG_DIR = "../dataset/images"
SAVE_PATH = "test.json"
imgs = glob.glob(f'{IMG_DIR}/*.png')
model = YOLO("best.pt")
j = json.load(open(f'../dataset/annotations/instances_default.json'))

images = j["images"]
for img_file in imgs:
    img = cv2.imread(img_file)
    result = model(img)[0]
    image = list(filter((lambda x: os.path.basename(x['file_name']) == os.path.basename(img_file)), images))
    if len(image) == 0:
        continue
    image = image[0]
    annotations = list(filter(lambda x : x["image_id"] == image["id"], j["annotations"]))
    
    if len(annotations) != 0:
        continue
    
    if result.masks == None:
        continue

    
    image_id = image['id']

    if result.masks != None:
        boxes = result.boxes  # 경계 상자 정보
        masks = result.masks  # 마스크 정보

        for box, mask in zip(boxes, masks):

            xy = np.array(mask.xy).astype(np.int32).tolist()

            flattened_points = [point for sublist in xy for point in sublist]
            
            x_coords = [point[0] for point in flattened_points]
            y_coords = [point[1] for point in flattened_points]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))
            bbox_x = top_left[0]
            bbox_y = top_left[1]
            bbox_w = bottom_right[0] - top_left[0]
            bbox_h = bottom_right[1] - top_left[1]
            area = bbox_w * bbox_h
            if area < 5000:
                continue

            cls_index = int(box.cls) 
            annotation = dict()
            annotation['id'] = len(j['annotations']) + 1
            annotation['image_id'] = image_id
            annotation['iscrowd'] = 0
            annotation['category_id'] = cls_index + 1
            attributes = dict()
            attributes['occluded'] = False
            annotation['attributes'] = attributes

            flattened_data = [[item for sublist in xy for pair in sublist for item in pair]]
            annotation['segmentation'] = flattened_data
            annotation['bbox'] = [
                bbox_x,
                bbox_y,
                bbox_w,
                bbox_h
            ]
            
            j["annotations"].append(annotation)

with open(SAVE_PATH, 'w') as json_file:          
    json.dump(j ,json_file)