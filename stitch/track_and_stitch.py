from ultralytics import YOLO
import os
import glob
import cv2


yolo11x = "/datadrive/codes/frank/ultralytics/run/runs/detect/train2/weights/best.pt"
model = YOLO(yolo11x)

results = model.track(source="/datadrive/codes/retail/ultralytics/stitch/data/103101.mp4", 
                      save=True,
                      tracker="/datadrive/codes/frank/ultralytics/ultralytics/cfg/trackers/bytetrack.yaml")

# imgs = glob.glob("/datadrive/codes/opensource/features/LightGlue/assets/pusi1/*.jpg")
# imgs = sorted(imgs,reverse=True)
# for i, img in enumerate(imgs):
#     results = model.track(source=img, 
#                         #   save=True,
#                         persist=True)
#     print(results[0].boxes.id)
#     results[0].save(filename=f"output_{i}.jpg")  # save image with bounding boxes

import json

for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    print("i", i, boxes.id)
    if i in [0, 21]:
        img = result.orig_img
        print("img", img.shape)
        cv2.imwrite(f"output_{i}.jpg", img)
        res_json = result.tojson()

        with open(f"result_{i}.json", "w") as f:
            json.dump(res_json, f)
    # print(boxes)
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # result.save(filename=f"output{i}.jpg")  # save image with bounding
    


# img0 = results[0].data.numpy()
# print(img0.shape)