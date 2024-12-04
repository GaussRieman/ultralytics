import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

name = "ccb7b3d2-f9c8-41aa-8f6a-cabe628bd928"

img_file = f"/datadrive/codes/retail/delfino/application/stitch/data/{name}.jpeg"


PROFILE_LABEL = "1056817"

img = cv2.imread(img_file)

# json_file = f"/datadrive/codes/retail/delfino/application/stitch/json/{name}.jpeg_res.json"
# seg_res = json.load(open(json_file))
# print("seg_res:", seg_res)
# h, w = img.shape[:2]


# profiles = []
# for i, polygon in enumerate(seg_res["polygons"]):
#     if seg_res["labels"][i] == PROFILE_LABEL:
#         profiles.append(polygon)
        
# profile = None
# max_area = 0
# if len(profiles) == 1:
#     profile = profiles[0]
# else:
#     for p in profiles:
#         points = p['points']
#         poly = [[pt['x']*w, pt['y']*h] for pt in points]
#         print("poly:", poly)
#         area = cv2.contourArea(np.array(poly, dtype=np.float32))
#         if area > max_area:
#             max_area = area
#             profile = p
# print("profile:", profile)


# points = profile['points']
# # find the actual four corners of the polygon, the corners are from the profile
# img_corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
# corners = []
# for img_corner in img_corners:
#     min_dist = 1e9
#     for point in points:
#         dist = (point['x'] - img_corner[0]) ** 2 + (point['y'] - img_corner[1]) ** 2
#         if dist < min_dist:
#             min_dist = dist
#             corner = [point['x']*w, point['y']*h]
#     corners.append(corner)
# print("corners:", corners)


# Manually set the corners
# corners = [[82,386], [1000, 375], [998, 424], [84, 464]]
# corners = [[0,348], [960, 362], [960, 823], [0, 848]] #5caeb
# corners = [[0, 520],[900, 541],[900, 779], [0, 777]] #406873ed-78ae-4a70-8218-c78cc0d60042

# # draw the corners, first convert the points to the image coordinates
# for corner in corners:
#     x = int(corner[0])
#     y = int(corner[1])
#     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
# cv2.imwrite(f"{name}_corners.jpg", img)

# # get the target corner points based on the profile corners
# x_coords = [corner[0] for corner in corners]
# y_coords = [corner[1] for corner in corners]
# min_x = min(x_coords)
# min_x = max(0, min_x)
# max_x = max(x_coords)
# min_y = min(y_coords)
# min_y = max(0, min_y)
# max_y = max(y_coords)

# target_corners = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
# # target_corners = [[0, 0], [w, 0], [w, h], [0, h]]
# print("target_corners:", target_corners)

# src_pts = np.array(corners, dtype=np.float32)
# dst_pts = np.array(target_corners, dtype=np.float32)

# M,_ = cv2.findHomography(src_pts, dst_pts)
# print("M:", M)

# # calculate the warped image size
# h, w = img.shape[:2]
# boungding_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
# boungding_corners = np.array([boungding_corners])
# boungding_corners = cv2.perspectiveTransform(boungding_corners, M)
# boungding_corners = np.squeeze(boungding_corners)
# print("new corners:", boungding_corners)
# min_x = int(min(boungding_corners[:, 0]))
# max_x = int(max(boungding_corners[:, 0]))
# min_y = int(min(boungding_corners[:, 1]))
# max_y = int(max(boungding_corners[:, 1]))
# print("min_x, max_x, min_y, max_y:", min_x, max_x, min_y, max_y)

# # calculate the translation to avoid negative coordinates
# Trans = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# if min_x < 0:
#     Trans[0, 2] = - min_x
#     max_x -= min_x
#     min_x = 0
    
# if min_y < 0:
#     Trans[1, 2] = - min_y
#     max_y -= min_y
#     min_y = 0

# new_h = max_y - min_y
# new_w = max_x - min_x

# M = np.dot(Trans, M)
# img_out = cv2.warpPerspective(img, M, (new_w, new_h))

# cv2.imwrite(f"{name}.jpg", img_out)



def rectify_image(image:np.ndarray, corners:np.ndarray):
    print("corners: ", corners, image.shape)
    if corners is None:
        print("No corners found")
        return image
    # Assertion
    if corners.shape[0] != 4:
        print("Invalid corners")
        return image
    print("Processing image")
    
    # get the target corner points based on the profile corners
    x_coords = [corner[0] for corner in corners]
    y_coords = [corner[1] for corner in corners]
    min_x = min(x_coords)
    min_x = max(0, min_x)
    max_x = max(x_coords)
    min_y = min(y_coords)
    min_y = max(0, min_y)
    max_y = max(y_coords)

    target_corners = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    print("target_corners:", target_corners)
    
    for p in corners:
        cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
        
    for p in target_corners:
        cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
        
    cv2.imwrite("corners.jpg", image)

    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array(target_corners, dtype=np.float32)
    M,_ = cv2.findHomography(src_pts, dst_pts)

    # calculate the warped image size
    h, w = image.shape[:2]
    boungding_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    boungding_corners = np.array([boungding_corners])
    boungding_corners = cv2.perspectiveTransform(boungding_corners, M)
    boungding_corners = np.squeeze(boungding_corners)

    min_x = int(min(boungding_corners[:, 0]))
    max_x = int(max(boungding_corners[:, 0]))
    min_y = int(min(boungding_corners[:, 1]))
    max_y = int(max(boungding_corners[:, 1]))

    # calculate the translation to avoid negative coordinates
    Trans = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if min_x < 0:
        Trans[0, 2] = - min_x
        max_x -= min_x
        min_x = 0
        
    if min_y < 0:
        Trans[1, 2] = - min_y
        max_y -= min_y
        min_y = 0

    new_h = max_y - min_y
    new_w = max_x - min_x

    M = np.dot(Trans, M)
    img_transformed = cv2.warpPerspective(img, M, (new_w, new_h))
    return img_transformed



def process_model_results(boxes:np.ndarray, points:np.ndarray, img:np.ndarray):
    bbox = None
    pts = None
    if len(boxes) == 0:
        return img, None, None

    # print("boxes: ", type(boxes))
    # print("points: ", type(points))
    if len(boxes) == 1:
        # print("boxes[0]: ", boxes[0])
        bbox = boxes[0]
        pts = points[0]
    else:
        max_area = 0
        for i, box in enumerate(boxes):
            # print("box: ", box)
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                bbox = box
                pts = points[i]
        print("max_area: ", max_area)
    cv2.imwrite("bbox.jpg", img)
    return img, np.array(bbox), np.array(pts)



if __name__ == "__main__":
    image_file = "/datadrive/codes/retail/ultralytics/datasets/test/abi/ccb7b3d2-f9c8-41aa-8f6a-cabe628bd928.png"
    img = cv2.imread(image_file)
    print("img: ", img.shape)
    
    # Load a pretrained YOLO11n-pose Pose model
    model = YOLO("/datadrive/codes/retail/ultralytics/runs/pose/train10/weights/best.onnx")

    # Run inference on an image
    results = model(image_file, device = "CPU")  # results list of detections
    r = results[0]  # first result
    print("type: ", type(r.keypoints.numpy().xy))

    # Get the keypoints
    kpts = r.keypoints.cpu().numpy().xy
    # print("kpts: ", type(kpts))
    bboxes = r.boxes.cpu().numpy().xyxy
    # print("bbox: ", type(bboxes))
    
    img, box, pts = process_model_results(bboxes, kpts, img)
    # print("box: ", box)
    # print("pts: ", pts)
    
    
    img_out = rectify_image(img, pts)
    cv2.imwrite("test.jpg", img_out)