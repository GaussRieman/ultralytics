# In this script, we will prepare the data for keypoint detection task.
# Input a csv with columns: ImgUrl,Polygon,ProductId, output a dataset for yolo11 pose training.


import json
import cv2
import numpy as np
import pandas as pd
import os
import shutil
import random
import sys
import requests
import retrying
import tqdm


PROFILE_LABEL = "1056817"
COLUMN_LABEL = "4433221"


@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def download_image(url, img_foler):
    os.makedirs(img_foler, exist_ok=True)
    # TODO:FileSCV has no extension
    img_name = url.split("/")[-1] + ".jpg"
    img_path = os.path.join(img_foler, img_name)
    if not os.path.exists(img_path):
        img = requests.get(url).content
        with open(img_path, 'wb') as f:
            f.write(img)
    return img_path



def polygon_to_four(points):
    img_corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    four = []
    bbox = []
    for img_corner in img_corners:
        min_dist = 1e9
        for point in points:
            dist = (point[0] - img_corner[0]) ** 2 + (point[1] - img_corner[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                corner = [point[0], point[1]]
        four.append(corner)
    min_x = min(four[0][0], four[1][0], four[2][0], four[3][0])
    max_x = max(four[0][0], four[1][0], four[2][0], four[3][0])
    min_y = min(four[0][1], four[1][1], four[2][1], four[3][1])
    max_y = max(four[0][1], four[1][1], four[2][1], four[3][1])
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    box_width = max_x - min_x
    box_height = max_y - min_y
    bbox = [center_x, center_y, box_width, box_height]
    return four, bbox



def select_polygon(polygons:list, labels:list, target_label:str=COLUMN_LABEL):
    to_select = []
    target_polygon = None
    for i, polygon in enumerate(polygons):
        if str(labels[i]) == target_label:
            to_select.append(polygon)
    if len(to_select) == 1:
        return to_select[0]
    else:
        max_area = 0
        for p in to_select:
            area = cv2.contourArea(np.array(p, dtype=np.float32))
            if area > max_area:
                max_area = area
                target_polygon = p
                
    return target_polygon



def process_dataframe(df:pd.DataFrame, save_folder:str):
    # create the save folder
    os.makedirs(save_folder, exist_ok=True)
    
    images_folder = os.path.join(save_folder, "images")
    train_images_folder = os.path.join(images_folder, "train")
    val_images_folder = os.path.join(images_folder, "val")
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    
    labels_folder = os.path.join(save_folder, "labels")
    train_labels_folder = os.path.join(labels_folder, "train")
    val_labels_folder = os.path.join(labels_folder, "val")
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    
    # random split the dataframe into train and val, the ratio is 0.9:0.1
    imgs = df["ImgUrl"].unique()
    random.shuffle(imgs)
    
    train_ration = 0.9
    if len(imgs) < 10:
        train_ration = 0.5
    train_imgs = imgs[:int(len(imgs)*train_ration)]
    val_imgs = imgs[int(len(imgs)*train_ration):]
    print(f"train_imgs:{len(train_imgs)}", f"val_imgs:{len(val_imgs)}")
    
    # process the dataframe
    count = 0
    stage = ["train", "val"]
    for st in stage:
        print(f"Processing {st} data")
        if st == "train":
            images_folder = train_images_folder
            labels_folder = train_labels_folder
            st_imgs = train_imgs
        else:
            images_folder = val_images_folder
            labels_folder = val_labels_folder
            st_imgs = val_imgs
        
        for img_url in tqdm.tqdm(st_imgs):
            # print(f"Processing {img_url}")
            try:
                img_path = download_image(img_url, images_folder)
                base_name = os.path.basename(img_path)
                
                # get the polygons and labels
                df_img = df[df["ImgUrl"] == img_url]
                polygons = df_img["Polygon"].values
                polygons = [eval(p) for p in polygons]
                labels = df_img["ProductId"].values
                # print(f"polygons:{polygons}", f"labels:{labels}")
                target_polygon = select_polygon(polygons, labels)
                # print(f"target_polygon:{target_polygon}")
                
                # save the label file
                corners, bbox = polygon_to_four(target_polygon)
                # print(f"bbox:{bbox}")
                label_path = os.path.join(labels_folder, base_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write("0")
                    for x in bbox:
                        f.write(f" {x}")
                    for pt in corners:
                        f.write(f" {pt[0]} {pt[1]}")
                count += 1
            except Exception as e:
                print(f"Error:{e}")
                continue
    
    print(f"Processed {count} images.")
    return True
        
        
    


def test_download_image():
    url = "https://fileman.clobotics.cn/api/file/ab9876c65bfa3557b03b700fa31af560"
    img_foler = "/datadrive/codes/retail/ultralytics/datasets/abi-stacking-pose/images/train"
    img_path = download_image(url, img_foler)
    print("img_path:", img_path)
    assert os.path.exists(img_path)
    # os.remove(img_path)
    
    

def process_csv_data(csv_file:str, save_folder:str):
    df = pd.read_csv(csv_file)
    result = process_dataframe(df, save_folder)
    return result



def visualize_data(data_folder:str):
    visualize_folder = os.path.join(data_folder, "visualize")
    os.makedirs(visualize_folder, True)
    stage = ["train", "val"]
    train_img_folder = os.path.join(visualize_folder, "images", "train")
    val_img_folder = os.path.join(visualize_folder, "images", "val")
    train_label_folder = os.path.join(visualize_folder, "labels", "train")
    val_label_folder = os.path.join(visualize_folder, "labels", "val")
    
    img_folder = None
    label_folder = None
    for st in stage:
        if st == "train":
            img_folder = train_img_folder
            label_folder = train_label_folder
        else:
            img_folder = val_img_folder
            label_folder = val_label_folder
        
        img_files = os.listdir(img_folder)
        for img_file in img_files:
            print(f"Processing {img_file}")
            img_path = os.path.join(img_folder, img_file)
            label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))
            with open(label_path, "r") as f:
                line = f.readline()
                line = line.strip().split(" ")
                # print(f"line:{line}")
                line = [eval(x) for x in line]
                bbox = line[1:5]
                corners = line[5:]
                img = cv2.imread(img_path)
                h,w,_ = img.shape
                center_x, center_y, box_width, box_height = bbox
                x0 = center_x - box_width/2
                y0 = center_y - box_height/2
                x1 = center_x + box_width/2
                y1 = center_y + box_height/2
                x0, x1 = int(x0*w), int(x1*w)
                y0, y1 = int(y0*h), int(y1*h)
                for i in range(4):
                    corners[i*2] = int(corners[i*2] * w)
                    corners[i*2+1] = int(corners[i*2+1] * h)
                print("corners:", corners)
                img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                for i in range(4):
                    img = cv2.circle(img, (corners[i*2], corners[i*2+1]), 5, (0, 0, 255), -1)
                cv2.imwrite(f"/datadrive/codes/retail/ultralytics/datasets/abi-stacking-pose/{st}_{img_file}", img)
        

if __name__ == "__main__":
    test_download_image()
    print("test_download_image passed")
    
    csv_file = "/datadrive/codes/retail/ultralytics/datasets/abi-stacking-pose/114384_f67158a8-53a2-447e-b94a-d93de9476bed_result.csv"
    save_folder = "/datadrive/codes/retail/ultralytics/datasets/abi-stacking-pose"
    process_csv_data(csv_file, save_folder)
    
    visualize_data(save_folder)