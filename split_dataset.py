import os
import random
import shutil

def split_dataset(base_path, train_ratio=0.8):
    images_dir = os.path.join(base_path, 'images')
    labels_dir = os.path.join(base_path, 'labels')
    
    # Создаем новые папки
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)
    
    # Получаем список всех изображений
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    def move_files(files, split):
        for f in files:
            # Копируем изображение
            shutil.copy2(os.path.join(images_dir, f), os.path.join(base_path, split, 'images', f))
            
            # Копируем соответствующую разметку
            label_file = os.path.splitext(f)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(base_path, split, 'labels', label_file))
            else:
                print(f"Warning: Label not found for {f}")

    move_files(train_images, 'train')
    move_files(val_images, 'val')
    print(f"Split complete: {len(train_images)} train, {len(val_images)} val.")

if __name__ == '__main__':
    split_dataset('/home/ubuntu/work/drake_repo/yolo_dataset')
