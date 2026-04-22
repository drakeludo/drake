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
        batch=64,            # МАКСИМАЛЬНЫЙ РАЗГОН: Загружаем 15ГБ VRAM
        workers=8,           # МАКСИМАЛЬНЫЙ РАЗГОН: Загружаем CPU на полную
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
    
    # Авто-загру    # 4. Экспорт в ONNX
    print("Экспортируем модель в формат ONNX...")
    try:
        onnx_path = model.export(format='onnx', imgsz=640)
        print(f"Модель успешно экспортирована: {onnx_path}")
    except Exception as e:
        print(f"Ошибка при экспорте в ONNX: {e}")

    # 5. Сохраняем результат на GitHub
    print("Загружаем результаты на GitHub...")
    os.system('git config --global user.email "colab@example.com"')
    os.system('git config --global user.name "Colab Trainer"')
    
    # Копируем веса и ONNX в корень для пуша
    os.system('cp runs/detect/orange_bot/weights/best.pt ./best.pt')
    if os.path.exists('runs/detect/orange_bot/weights/best.onnx'):
        os.system('cp runs/detect/orange_bot/weights/best.onnx ./best.onnx')
    
    os.system('git add best.pt best.onnx')
    os.system('git commit -m "Update trained model weights (PT and ONNX)"')
    os.system('git push origin main')    print("--- Веса успешно обновлены в репозитории! ---")

if __name__ == '__main__':
    train()
