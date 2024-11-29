from ultralytics import YOLO
import os
import glob
import cv2
import json
import numpy as np
import loguru

logger = loguru.logger
logger.add("stitch_task.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")

def get_boundingBox(Hs, corners):
    ## Get offset and size
    pts = []
    for H in Hs:
        warp_corners = cv2.perspectiveTransform(corners, H)
        pts.append(warp_corners)
    pts = np.concatenate(pts)
    rect = cv2.boundingRect(pts)
    print("rect", rect)
    return rect


def verify_corners(corners):
    corners = corners.reshape(4, 2)
    _, _, w, h = cv2.boundingRect(corners)
    if min(w, h) < 1: return -1
    if not cv2.isContourConvex(corners): return -1
    if corners[0][0] >= corners[1][0]: logger.warning(f"Top left X is greater than top right X")
    if corners[3][0] >= corners[2][0]: logger.warning(f"Bottom left X is greater than bottom right X")
    if corners[0][1] >= corners[3][1]: logger.warning(f"Top left Y is greater than bottom left Y")
    if corners[1][1] >= corners[2][1]: logger.warning(f"Top right Y is greater than bottom right Y")
    area = cv2.contourArea(corners)
    return area


def adjust_by_pov(Hs, corners):
    best_pov = None
    best_score = 0

    logger.info('Adjusting PoV ...')
    for refH in Hs:
        areas = []
        pov = np.linalg.inv(refH)
        for H in Hs:
            warp_corners = cv2.perspectiveTransform(corners, np.dot(pov, H))
            area = verify_corners(warp_corners)
            if area < 0:
                areas = []
                break
            areas.append(area)
        if not areas:
            continue
        score = min(areas) / max(areas)
        if best_pov is None or score > best_score:
            best_score = score
            best_pov = pov

    logger.info(f'==> Best PoV score: {best_score:.3f}')
    if (best_score <= 0.01):
        logger.warning(f'ERROR: Low best pov score!')
        # sys.exit('ERROR: Low best pov score!')

    if best_pov is None:
        return [H.copy() for H in Hs]
    else:
        return [np.dot(best_pov, H) for H in Hs]



def adjust_roi(homographies, corners, max_size):

    ## Get offset and size
    sl, st, sw, sh = get_boundingBox(homographies, corners)
    logger.info(f'Adjusting RoI of panorama ...{sl, st, sh, sw}')

    ## Rescale
    s = min(max_size / max(sw, sh), 1)
    offset = np.float32([s, 0, -s * sl, 0, s, -s * st, 0, 0, 1]).reshape(3, 3)
    homographies = [np.dot(offset, H) for H in homographies]
    dl, dt, dw, dh = get_boundingBox(homographies, corners)

    logger.info(f'==> Rescaled from {sw}x{sh} to {dw}x{dh}')
    logger.info(f'==> Offset from ({sl}, {st}) to ({dl}, {dt})')

    return homographies, dl, dt, dw, dh



def rectify_horizontally(homographies, corners):
    ## Get edges
    left_pts, right_pts = [], []
    for H in homographies:
        warp_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        left_pts.extend([warp_corners[0], warp_corners[3]])
        right_pts.extend([warp_corners[1], warp_corners[2]])
    left_pts.sort(key=lambda v: v[0])
    right_pts.sort(key=lambda v: v[0])


    ## Get size
    l, t, w, h = get_boundingBox(homographies, corners)


    ## Get left edges: x = ay + b
    num = len(left_pts)
    left_edges = []
    for i1 in range(num):
        x1, y1 = left_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = left_pts[i2]
            if y1 == y2:
                continue
            a = (x1 - x2) / (y1 - y2)
            b = x1 - a * y1
            valid = True
            for x, y in left_pts + right_pts:
                if round(x) < round(y * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            xt = a * t + b
            xb = a * (t + h) + b
            left_edges.append((xt, xb))


    ## Get bottom edges: x = ay + b
    num = len(right_pts)
    right_edges = []
    for i1 in range(num):
        x1, y1 = right_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = right_pts[i2]
            if y1 == y2:
                continue
            a = (x1 - x2) / (y1 - y2)
            b = x1 - a * y1
            valid = True
            for x, y in left_pts + right_pts:
                if round(x) > round(y * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            xt = a * t + b
            xb = a * (t + h) + b
            right_edges.append((xt, xb))


    ## Find smallest quadrangle
    r,  b = l + w, t + h
    polys = []
    for tl, bl in left_edges:
        for tr, br in right_edges:
            if tl >= tr or bl >= br:
                continue
            poly = np.float32([tl, t, tr, t, br, b, bl, b]).reshape(4, 1, 2)
            area = cv2.contourArea(poly)
            polys.append((area, (tl, bl, tr, br)))
    if not polys:
        return homographies
    polys.sort(key=lambda x: x[0])
    area, poly = polys[0]
    if area > w * h:
        return homographies


    ## Align both edges
    tl, bl, tr, br = poly
    l, r = 0.5 * (tl + bl), 0.5 * (tr + br)
    src_pts = np.float32([tl, t, tr, t, br, b, bl, b]).reshape(4, 2)
    dst_pts = np.float32([l, t, r, t, r, b, l, b]).reshape(4, 2)
    T = cv2.getPerspectiveTransform(src_pts, dst_pts)
    homographies = [np.dot(T, H) for H in homographies]
    return homographies

def rectify_vertically(homographies, corners):


    ## Get edges
    top_pts, bottom_pts = [], []
    for H in homographies:
        warp_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        top_pts.extend([warp_corners[0], warp_corners[1]])
        bottom_pts.extend([warp_corners[3], warp_corners[2]])
    top_pts.sort(key=lambda v: v[1])
    bottom_pts.sort(key=lambda v: v[1])


    ## Get size
    l, t, w, h = get_boundingBox(homographies, corners)


    ## Get top edges: y = ax + b
    top_edges = []
    num = len(top_pts)
    for i1 in range(num):
        x1, y1 = top_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = top_pts[i2]
            if x1 == x2:
                continue
            a = (y1 - y2) / (x1 - x2)
            b = y1 - a * x1
            valid = True
            for x, y in top_pts + bottom_pts:
                if round(y) < round(x * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            yl = a * l + b
            yr = a * (l + w) + b
            top_edges.append((yl, yr))


    ## Get bottom edges: y = ax + b
    num = len(bottom_pts)
    bottom_edges = []
    for i1 in range(num):
        x1, y1 = bottom_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = bottom_pts[i2]
            if x1 == x2:
                continue
            a = (y1 - y2) / (x1 - x2)
            b = y1 - a * x1
            valid = True
            for x, y in top_pts + bottom_pts:
                if round(y) > round(x * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            yl = a * l + b
            yr = a * (l + w) + b
            bottom_edges.append((yl, yr))


    ## Find smallest quadrangle
    r,  b = l + w, t + h
    polys = []
    for tl, tr in top_edges:
        for bl, br in bottom_edges:
            if tl >= bl or tr >= br:
                continue
            poly = np.float32([l, tl, r, tr, r, br, l, bl]).reshape(4, 1, 2)
            area = cv2.contourArea(poly)
            polys.append((area, (tl, tr, bl, br)))
    if not polys:
        return homographies
    polys.sort(key=lambda x: x[0])
    area, poly = polys[0]
    if area > w * h:
        return homographies


    ## Align both edges
    tl, tr, bl, br = poly
    t, b = 0.5 * (tl + tr), 0.5 * (bl + br)
    src_pts = np.float32([l, tl, r, tr, r, br, l, bl]).reshape(4, 2)
    dst_pts = np.float32([l, t, r, t, r, b, l, b]).reshape(4, 2)
    T = cv2.getPerspectiveTransform(src_pts, dst_pts)
    homographies = [np.dot(T, H) for H in homographies]
    return homographies



### 1.Get the detected images and tracking results
def get_tracking_results():
    yolo11x = "/datadrive/codes/frank/ultralytics/run/runs/detect/train2/weights/best.pt"
    model = YOLO(yolo11x)

    results = model.track(source="/datadrive/codes/retail/ultralytics/stitch/data/demo1.MOV", 
                        save=True,
                        tracker="/datadrive/codes/frank/ultralytics/ultralytics/cfg/trackers/bytetrack.yaml")

    # save the detected images and tracking results
    img_output_folder = "/datadrive/codes/retail/ultralytics/stitch/output/imgs"
    json_output_folder = "/datadrive/codes/retail/ultralytics/stitch/output/jsons"
    os.makedirs(img_output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    sep = 5
    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        print("i", i, boxes.id)
        if i % sep == 0:
            img = result.orig_img
            print("img", img.shape)
            f_name = os.path.join(img_output_folder, f"{i}.jpg")
            cv2.imwrite(f_name, img)
            
            res_json = result.to_json()
            json_name = os.path.join(json_output_folder, f"{i}.json")
            with open(json_name, "w") as f:
                # json.dump(res_json, f)
                f.write(res_json)
                

### 2. Stitch the detected images
# custom sorting function  
def sort_key_func(item):  
    base_name = os.path.basename(item)  # get file name with extension  
    num = os.path.splitext(base_name)[0]  # remove extension  
    return int(num)  


#stitch by homography
def stitch_by_homographies(imgs:np.ndarray, homographies:np.ndarray):
    h, w = imgs[0].shape[:2]
    print("h: ", h, "w: ", w)
    Hs = homographies
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
    # print("x_min: ", x_min, "x_max: ", x_max, "y_min: ", y_min, "y_max: ", y_max)  

    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    for i in range(len(Hs)):  
        Hs[i] = np.dot(translation, Hs[i]) 

    width = xmax - xmin  
    height = ymax - ymin  
    print("width: ", width, "height: ", height)  
    pano = np.zeros((height, width, 3), np.uint8)

    # add to pano from left to right
    for img, H in zip(imgs, Hs):
        # print("H: ", H)
        cv2.warpPerspective(img, H, (width, height), pano, borderMode=cv2.BORDER_TRANSPARENT)
    
    return pano


def do_stitch(imgs:np.ndarray, Hs:np.ndarray):
    h,w = imgs[0].shape[:2]
    corners = np.float32([0, 0, w, 0, w, h, 0, h]).reshape(4, 1, 2)
    Hs = adjust_by_pov(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    Hs = rectify_horizontally(Hs, corners)
    Hs = rectify_vertically(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    
    ## Generate panorama
    print("l, t, pw, ph", pw, ph)
    # assert abs(l) < 2 and abs(t) < 2, f'BUG #1 in stitch(): {l}, {t}'
    pano = np.zeros((ph, pw, 3), np.uint8)

    for img, H in zip(imgs, Hs):
        cv2.warpPerspective(img, H, (pw, ph), pano, borderMode=cv2.BORDER_TRANSPARENT)
        
    return pano


def cv_stitch(imgs:np.ndarray):
    # Create a Stitcher class object  
    stitcher = cv2.Stitcher.create()  
    
    # Pass the images to the stitch method  
    status, panorama = stitcher.stitch(imgs)  
    
    if status == cv2.Stitcher_OK:  
        # Save the resulting image  
        cv2.imwrite('result.jpg', panorama)  
    else:  
        print('Error during stitching, error code: ', status)  
    return panorama


def stitch_video():
    frm_cnt = len(glob.glob("/datadrive/codes/retail/ultralytics/stitch/output/imgs/*.jpg"))
    conf_thresh = 0.6
    boun_thresh = 0.001
    img_folder = "/datadrive/codes/retail/ultralytics/stitch/output/imgs"
    json_folder = "/datadrive/codes/retail/ultralytics/stitch/output/jsons"
    
    imgs = glob.glob(os.path.join(img_folder, "*.jpg"))
    imgs = sorted(imgs, key=sort_key_func)

    jsons = glob.glob(os.path.join(json_folder, "*.json"))
    jsons = sorted(jsons, key=sort_key_func)

    stitch_imgs = imgs[:frm_cnt]
    stitch_jsons = jsons[:frm_cnt]
    
    #select the keypoints
    v_key_points = []
    v_tracking_ids = []
    v_imgs = []
    for img_file, json_file in zip(stitch_imgs[:frm_cnt], stitch_jsons[:frm_cnt]):
        img = cv2.imread(img_file)
        h,w,_ = img.shape
        x_bound = [w*x for x in [boun_thresh, 1-boun_thresh]]
        y_bound = [h*x for x in [boun_thresh, 1-boun_thresh]]
        v_imgs.append(img)
        # print("x_bound", x_bound)
        # print("y_bound", y_bound)
        
        with open(json_file, "r") as read_file:
            tracking_ids = []
            key_points = []
            data = json.load(read_file)
            for d in data:
                conf = d['confidence']
                if conf > conf_thresh:
                    x1, y1 = d['box']['x1'], d['box']['y1']
                    x2, y2 = d['box']['x2'], d['box']['y2']
                    if x1 < x_bound[0] or x2 > x_bound[1] or y1 < y_bound[0] or y2 > y_bound[1]:
                        print("out of bound", x1, y1, x2, y2)
                        continue
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    key_points.append(cv2.KeyPoint(x1, y1, 1))
                    key_points.append(cv2.KeyPoint(x2, y2, 1))
                    tracking_ids.append(d['track_id'])
            
            v_key_points.append(key_points)
            v_tracking_ids.append(tracking_ids)
    
    print("v_key_points", len(v_key_points[0]))
    print("v_tracking_ids", len(v_tracking_ids[0]))
    
    # Find homography
    v_homographies = []
    for i in range(1, frm_cnt):
        cur_ids = v_tracking_ids[i]
        pre_ids = v_tracking_ids[i-1]
        common_ids = list(set(cur_ids) & set(pre_ids))
        src_points = []
        dst_points = []
        for j in range(len(cur_ids)):
            if cur_ids[j] in common_ids:
                print("common_ids", cur_ids[j], v_key_points[i][j])
                src_points.append(v_key_points[i][2*j])
                src_points.append(v_key_points[i][2*j+1])
        
        for k in range(len(pre_ids)):
            if pre_ids[k] in common_ids:
                print("common_ids", pre_ids[k], v_key_points[i-1][k])
                dst_points.append(v_key_points[i-1][2*k])
                dst_points.append(v_key_points[i-1][2*k+1])
            
        src_points = np.array([p.pt for p in src_points], dtype=np.float32)
        dst_points = np.array([p.pt for p in dst_points], dtype=np.float32)
        print("src_points", src_points)
        print("dst_points", dst_points)
        
        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        print("M", M)
        v_homographies.append(M)
        
    Hs = [np.eye(3, 3, dtype=np.float32)]
    for i in range(len(v_homographies)):
        H = np.dot(Hs[-1], np.float32(v_homographies[i]).reshape(3, 3))
        Hs.append(H)
    
    print("v_homographies", Hs)
    pano = do_stitch(v_imgs, Hs)
    cv2.imwrite("pano.jpg", pano)

if __name__ == "__main__":
    get_tracking_results()
    
    # stitch_video()
    
    # imgs = [cv2.imread(f) for f in glob.glob("/datadrive/codes/retail/ultralytics/stitch/output_near/imgs/*.jpg")]
    # pano = cv_stitch(imgs)
    # print("pano", pano.shape)
    # cv2.imwrite("pano_cv.jpg", pano)
    