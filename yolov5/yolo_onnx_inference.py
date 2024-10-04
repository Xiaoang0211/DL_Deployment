import onnxruntime
import cv2
import numpy as np
import time

# Load ONNX model
model_path = "yolov5s.onnx"

# Create InferenceSession with GPU execution provider
session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Load class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Preprocess frame
def preprocess(frame):
    # Resize frame to model input size
    img_resized = cv2.resize(frame, (640, 640))
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img_normalized = img_rgb / 255.0
    # Transpose dimensions to (C, H, W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)
    # Add batch dimension
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched

# Post-process output
def post_process(output, conf_threshold=0.4, iou_threshold=0.5):
    predictions = output[0]
    # Reshape predictions if needed
    if len(predictions.shape) == 3:
        predictions = np.squeeze(predictions, axis=0)
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_probs = predictions[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = scores * class_probs[np.arange(len(class_probs)), class_ids]

    # Filter out low-confidence detections
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    # Convert boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Scale boxes to original frame size
    boxes_xyxy *= [original_width / 640, original_height / 640, original_width / 640, original_height / 640]

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_threshold, iou_threshold
    )
    if len(indices) > 0:
        indices = indices.flatten()
        boxes_xyxy = boxes_xyxy[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
    else:
        boxes_xyxy = np.array([])
        confidences = np.array([])
        class_ids = np.array([])

    return boxes_xyxy, confidences, class_ids

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize variables for FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess frame
    img_batched = preprocess(frame)

    # Run inference
    input_name = session.get_inputs()[0].name
    time_before_inference = time.time()
    output = session.run(None, {input_name: img_batched})
    time_after_inference = time.time()
    # Post-process output
    boxes, confidences, class_ids = post_process(output)

    # Draw detections on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        confidence = confidences[i]
        class_id = class_ids[i]
        label = f"{class_names[class_id]}: {confidence:.2f}"

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    fps = 1 / (time_after_inference - time_before_inference)

    # Display FPS on frame
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Display the resulting frame
    cv2.imshow("YOLOv5 ONNX Webcam Inference", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
