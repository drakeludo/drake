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
        epochs=2000,         # Увеличено до 2000 эпох для максимальной точности
        imgsz=640,
        batch=32,            # Увеличено для максимальной загрузки GPU
        workers=4,           # Увеличено для максимальной загрузки CPU
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
    
    # Авто-загрузка на GitHub
    print("--- Отправка результатов на GitHub... ---")
    os.system('git config --global user.email "colab@google.com"')
    os.system('git config --global user.name "Google Colab"')
    os.system('cp runs/detect/orange_bot/weights/best.pt ./best.pt')
    os.system('git add best.pt')
    os.system('git commit -m "feat: Update best.pt after 2000 epochs training in Colab"')
    os.system('git push origin main')
    print("--- Веса успешно обновлены в репозитории! ---")

if __name__ == '__main__':
    train()
