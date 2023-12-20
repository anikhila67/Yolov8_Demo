from ultralytics import YOLO

model_path = r"E:\solar-v9\train6\weights\best.pt"

model = YOLO(model_path) 
model.export(format="onnx", imgsz=[480,640])