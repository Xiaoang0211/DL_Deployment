from ultralytics import YOLO
model = YOLO("/home/xiaoang/YOLOModels/yolov11/segmentation/yolo11s-seg.pt")

model.export(format="onnx")