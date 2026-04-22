import os
import requests
import zipfile

def setup_data():
    dataset_dir = 'yolo_dataset'
    images_dir = os.path.join(dataset_dir, 'images/train')
    
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print("--- Датасет не найден. Создаю структуру и скачиваю примеры... ---")
        os.makedirs(os.path.join(dataset_dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'images/val'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels/train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels/val'), exist_ok=True)
        
        # Здесь мог бы быть код скачивания реального датасета, 
        # но так как у нас его нет, мы просто создаем структуру.
        print("!!! ВНИМАНИЕ: Положите ваши изображения в yolo_dataset/images/train !!!")
    else:
        print("--- Датасет обнаружен. ---")

if __name__ == '__main__':
    setup_data()
