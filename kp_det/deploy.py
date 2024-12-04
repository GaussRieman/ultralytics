from ultralytics import YOLO
from ultralytics.models.yolo.pose.predict import PosePredictor
import cv2

img = "/datadrive/codes/retail/ultralytics/datasets/abi-stacking-1202/images/val/0b21ced7fdd84cb699428318016860d0.jpg"

# Load a model
model_path = "/datadrive/codes/retail/ultralytics/runs/pose/train10/weights/best.pt"
model = YOLO(model_path)

# Export the model
model.export(format="onnx", 
             imgsz=320, 
             opset=12, 
             dynamic=True)


# Local onnx model
# onnx_model = YOLO("/datadrive/codes/retail/ultralytics/deploy/plane_detection/1/model.onnx", 
#                   task="pose")
# results = onnx_model(img, imgsz=640, save=True)
# print("Results: ", results)



# Triton onnx model
# model = YOLO("http://localhost:8003/plane_detection/1", task="pose")
# print("Model: ", Model.is_triton_model("http://localhost:8003/plane_detection/"))
# results = model(img, imgsz=640, save=True)
# print("Results: ", results)




# Triton server
# from tritonclient.grpc import service_pb2_grpc, service_pb2, model_config_pb2,InferInput, InferRequestedOutput
# from tritonclient.grpc import InferenceServerClient
# import tritonclient.http as httpclient


# with InferenceServerClient(url="localhost:8004", verbose=False) as client:

#     # check the models
#     # print(client.get_model_repository_index())
#     # print(client.get_model_config("plane_detection", "1"))
#     # print(client.get_model_metadata("plane_detection", "1"))

#     # Infer
#     inputs = []
#     outputs = []
#     inputs.append(InferInput('images', [1, 3, 640, 640], "FP32"))
#     outputs.append(InferRequestedOutput('output0'))
    
#     # Load the image
#     import cv2
#     import numpy as np
#     img = cv2.imread(img)
#     img = cv2.resize(img, (640, 640))
#     img = img.transpose(2, 0, 1)
#     img = img[np.newaxis, ...]
#     img = img.astype(np.float32)
#     img = img / 255.0
#     print("img: ", img.shape)
#     inputs[0].set_data_from_numpy(img)
#     results = client.infer(model_name="plane_detection", inputs=inputs, outputs=outputs)
    
#     results = results.as_numpy("output0")
#     print("Results: ", results.shape)
    
