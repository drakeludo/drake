from ultralytics import YOLO
import os

dataset_path = 'yolo_dataset/dataset.yaml'
weights_path = 'best.pt'
model = YOLO(weights_path)

results = model.train(
    data=dataset_path,
    epochs=5000,
    imgsz=640,
    batch=-1,
    workers=8,
    name='orange_hardcore_640',
    exist_ok=True,
    augment=True,
    box=20.0,
    cls=2.0,
    dfl=3.0,
    patience=100,
    optimizer='AdamW',
    lr0=0.001
)
