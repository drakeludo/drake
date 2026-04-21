from ultralytics import YOLO
import os

# Путь к датасету
dataset_path = 'yolo_dataset/dataset.yaml'
# Путь к лучшим весам
weights_path = 'best.pt'

# Загружаем модель
model = YOLO(weights_path)

# УЛЬТРА-ХАРДКОРНОЕ ОБУЧЕНИЕ (20,000 ЭПОХ)
# box=25.0 - экстремальный штраф за промах (снайперская точность)
# imgsz=640 - максимальное разрешение для деталей
results = model.train(
    data=dataset_path,
    epochs=20000,   # 20 тысяч эпох для идеального результата
    imgsz=640,
    batch=-1,       # Авто-подбор батча под GPU
    workers=8,
    name='orange_hardcore_20k',
    exist_ok=True,
    augment=True,   # Максимальная аугментация для разных условий
    box=25.0,       # Максимальный фокус на центр апельсина
    cls=2.0,
    dfl=3.0,
    patience=500,   # Увеличиваем терпение для долгого обучения
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True     # Косинусное затухание для плавной сходимости
)
