import time
import cv2
import torch
import numpy as np

# Import YOLOv5 module
import sys
sys.path.insert(0, './yolov5')  # add yolov5 to path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load YOLOv5 model
model_path = 'yolov5s.pt'  # Adjust the path if necessary

# Select device (GPU if available, else CPU)
device = select_device('0') 

# Load model
model = attempt_load(model_path, device=device)  # Adjusted here

# Set model to evaluation mode
model.eval()

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or specify the device index

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
    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.0  # Normalize to [0,1]

    # Resize and pad image to 640x640 for YOLOv5
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_transposed).to(device)
    img_tensor = img_tensor.float()
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        time_before_inference = time.time()
        pred = model(img_tensor)[0]
        time_after_inference = time.time()
    
    # Apply NMS
    conf_threshold = 0.25
    iou_threshold = 0.45
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)

    # Process detections
    for det in pred:
        if det is not None and len(det):
            # Rescale boxes from img_size to original frame size
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]}: {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    fps = 1 / (time_after_inference - time_before_inference)

    # Put FPS text on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Webcam', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
