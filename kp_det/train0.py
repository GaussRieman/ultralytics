from ultralytics import YOLO
from ultralytics.data.augment import RandomPerspective

# Load a model
# model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11m-pose.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/datadrive/codes/retail/ultralytics/datasets/abi-stitching-1303/abi-stacking-pose-1202.yaml", 
                      epochs=200, 
                      imgsz=320,
                      batch=64)


# data = 