import cv2
import torch
import time
import numpy as np

# Import YOLOv5 modules
import sys
sys.path.insert(0, './yolov5')  # add yolov5 to path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Initialize
model_path = 'yolov5s.engine'  # Path to your TensorRT engine file
device = select_device('0')     # Select GPU device

# Load model
model = DetectMultiBackend(model_path, device=device)
stride, pt = model.stride, model.pt
img_size = (640, 640)  # Input image size

# Load class names from coco.names file
with open('coco.names', 'r') as f:
    names = [line.strip() for line in f.readlines()]

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Start time to calculate FPS
    new_frame_time = time.time()

    # Prepare image
    img = cv2.resize(frame, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float()
    img_tensor /= 255.0  # Normalize to [0,1]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Inference
    time_before_inference = time.time()
    pred = model(img_tensor)
    time_after_inference = time.time()
    
    fps = 1/(time_after_inference - time_before_inference)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = frame.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]}: {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw rectangle
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(im0, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Put FPS text on frame
    cv2.putText(im0, f"FPS: {fps:.2f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 TensorRT', im0)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
