import cv2
import os
from ultralytics import YOLO

model = YOLO('best.pt')
img_dir = 'yolo_dataset/images/val'
output_dir = 'sniper_results'
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(img_dir)[:5]:
    img_path = os.path.join(img_dir, img_name)
    results = model(img_path)
    img = cv2.imread(img_path)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(img, "CLICK", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"sniper_{img_name}"), img)
