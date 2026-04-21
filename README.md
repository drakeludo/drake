# Majestic RP: Orange Collector AI 🍊

Этот репозиторий содержит всё необходимое для обучения нейросети, которая будет находить апельсины в мини-игре на Majestic RP.

## 🚀 Быстрый старт (для твоего ПК)

### 1. Подготовка окружения
Убедись, что у тебя установлен Python 3.10+ и драйвера NVIDIA.
```bash
pip install ultralytics torch torchvision torchaudio opencv-python
```

### 2. Запуск обучения
Просто запусти этот скрипт. Он сам найдет твою видеокарту и начнет обучение.
```bash
python train_optimized.py
```

### 3. Где забрать результат?
После завершения (или если нажмешь Ctrl+C), лучшие веса будут здесь:
- `runs/detect/majestic_orange_bot/weights/best.pt` (для Python)
- `runs/detect/majestic_orange_bot/weights/best.onnx` (для быстрых ботов на C++/C#)

## 📂 Структура проекта
- `yolo_dataset/` - подготовленный датасет (изображения + разметка)
- `train_optimized.py` - умный скрипт обучения с поддержкой GPU
- `yolov8n.pt` - базовая модель от Ultralytics

## 💡 Советы
- Если обучение идет слишком медленно, уменьши `imgsz` до 416 в файле `train_optimized.py`.
- Для лучшего результата не прерывай обучение хотя бы первые 200-300 эпох.

## 🛠 Продвинутая настройка

### Установка CUDA (для GPU)
Чтобы обучение летало, нужно:
1. Установить [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (рекомендуется v11.8 или v12.1).
2. Установить [cuDNN](https://developer.nvidia.com/cudnn).
3. Переустановить PyTorch с поддержкой CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Как сделать бота умнее?
- **Больше данных**: Добавляй свои скрины в `yolo_dataset/images` и размечай их в [LabelImg](https://github.com/HumanSignal/labelImg) или [Roboflow](https://roboflow.com).
- **Аугментация**: В папке `utils/` есть скрипты для создания "плохой погоды" — это поможет боту не тупить в дождь или ночью.
- **FPS**: Если бот тормозит, используй `best.onnx` через библиотеку `onnxruntime`.

## 📈 Метрики
Следи за `box_loss` и `cls_loss` в консоли. Если они падают — всё идет по плану. Если `mAP50` выше 0.9 — твой бот станет машиной для сбора апельсинов.
