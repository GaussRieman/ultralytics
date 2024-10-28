import json 
import cv2
import numpy as np



with open("/datadrive/codes/retail/ultralytics/stitch/output/jsons/30.json", "r") as read_file:  
    data21 = json.load(read_file)


with open("/datadrive/codes/retail/ultralytics/stitch/output/jsons/15.json", "r") as read_file:  
    data0 = json.load(read_file)


src_points = []
dst_points = []
for d in data21:
    if d['track_id'] in [1, 2, 4, 5, 6, 7]:
        x1, y1 = d['box']['x1'], d['box']['y1']
        x2, y2 = d['box']['x2'], d['box']['y2']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        src_points.append(cv2.KeyPoint(x1, y1, 1))
        src_points.append(cv2.KeyPoint(x2, y2, 1))
        
print("shape", len(src_points))
print("src_points", [p.pt for p in src_points])

for d in data0:
    if d['track_id'] in [1, 2, 4, 5, 6, 7]:
        x1, y1 = d['box']['x1'], d['box']['y1']
        x2, y2 = d['box']['x2'], d['box']['y2']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        dst_points.append(cv2.KeyPoint(x1, y1, 1))
        dst_points.append(cv2.KeyPoint(x2, y2, 1))
        
print("shape", len(dst_points))
print("dst_points", [p.pt for p in dst_points])


# Find homography  
src_points = np.array([p.pt for p in src_points], dtype=np.float32)
dst_points = np.array([p.pt for p in dst_points], dtype=np.float32)
M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
print("M", M)
  
  
img0_path = "/datadrive/codes/retail/ultralytics/stitch/output/imgs/15.jpg"
img1_path = "/datadrive/codes/retail/ultralytics/stitch/output/imgs/30.jpg"
# Use this homography  
#stitching  
img0 = cv2.imread(img0_path)  
img1 = cv2.imread(img1_path)  
h, w = img1.shape[:2]  
H0 = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)  
H1 = H0 @ M  
print("H1 \n", H1)
Hs = [H0, H1]  
x_min = 0  
x_max = 0  
y_min = 0  
y_max = 0  
for i, H in enumerate(Hs):  
    corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])  
    transformed_corners = cv2.perspectiveTransform(np.float32([corners]), H)  
    transformed_corners = np.int32(transformed_corners[0])  
    # print("transformed_corners:", transformed_corners)  
    x_min = min(x_min, min(transformed_corners[:, 0]))  
    x_max = max(x_max, max(transformed_corners[:, 0]))  
    y_min = min(y_min, min(transformed_corners[:, 1]))  
    y_max = max(y_max, max(transformed_corners[:, 1]))  

xmin = min(0, x_min)  
xmax = max(w, x_max)  
ymin = min(0, y_min)  
ymax = max(h, y_max)
print("x_min: ", x_min, "x_max: ", x_max, "y_min: ", y_min, "y_max: ", y_max)  

translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
for i in range(len(Hs)):  
    Hs[i] = np.dot(translation, Hs[i]) 

images = [img0, img1]  
width = xmax - xmin  
height = ymax - ymin  
print("width: ", width, "height: ", height)  
pano = np.zeros((height, width, 3), np.uint8)

# add to pano from left to right
for img, H in zip(images, Hs):
    # print("H: ", H)
    cv2.warpPerspective(img, H, (width, height), pano, borderMode=cv2.BORDER_TRANSPARENT)

#save result
cv2.imwrite(f"output_stitch.jpg", pano)