from ultralytics import YOLO
import os
import torch

def train():
    # 1. Пути
    dataset_path = os.path.abspath('yolo_dataset/dataset.yaml')
    
    # 2. Выбор модели
    # Если есть best.pt, продолжаем обучение, иначе берем базовую yolov8n
    model_path = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
    print(f"--- Используем модель: {model_path} ---")
    model = YOLO(model_path)

    # 3. Настройка параметров для Colab
    # Мы используем умеренные параметры, которые точно не "уронят" сессию
    results = model.train(
        data=dataset_path,
        epochs=100,          # 100 эпох для начала, можно увеличить
        imgsz=640,
        batch=16,            # Стабильный батч для T4 GPU
        workers=2,           # Оптимально для 2-ядерного CPU в Colab
        name='orange_bot',
        exist_ok=True,
        augment=True,        # Аугментация важна для игр
        box=7.5,             # Стандартный вес
        cls=0.5,
        dfl=1.5,
        patience=20,         # Остановимся, если 20 эпох нет улучшений
        optimizer='auto',
        project='runs/detect',
        save=True,
        save_period=10       # Сохраняем каждые 10 эпох на всякий случай
    )
    print("--- Обучение завершено! ---")

if __name__ == '__main__':
    train()
