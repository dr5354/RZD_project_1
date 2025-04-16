# Импорт библиотек
import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.distributed.checkpoint import load_state_dict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import json

# Задание констант (директория для сохранения предсказаний, путь к изображениям и модели)
pred = r'path_to_pred'
model_d = r'path_to_b_model'
Img_d = r'path_to_imgs'
Mean = [0.36149433, 0.36072833, 0.37245706]  # Вычисленные среднее и отклонение для датасета
Std = [0.17737652, 0.17860805, 0.20081734]
Score = 0.0001

# Классы датасета
un_cls = ['Car', 'FacingSwitchL', 'FacingSwitchNV', 'FacingSwitchR', 'Human',
                  'SignalE', 'SignalF', 'TrailingSwitchL', 'TrailingSwitchNV',
                  'TrailingSwitchR', 'Wagon']
un_cls.insert(0,'Background')
print(un_cls)
#n_cl = len(UNIQUE_CLASSES)+1 
i_c = {
    0: 'background', 1: 'Car', 2: 'FacingSwitchL', 3: 'FacingSwitchNV',
    4: 'FacingSwitchR', 5: 'Human', 6: 'SignalE', 7: 'SignalF',
    8: 'TrailingSwitchL', 9: 'TrailingSwitchNV', 10: 'TrailingSwitchR',
    11: 'Wagon'
} #Из датасета, вывели в консоль при обучении
# По возможности используем cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Предобработка изображений
inf_t = A.Compose([
    A.Resize(800, 800),
    A.Normalize(mean=Mean, std=Std),
    ToTensorV2(),
])
n_cl = 12
model = FasterRCNN(resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V2),num_classes=n_cl,nms=0.3)#Нужен более кастомный нмс
model.load_state_dict(torch.load(model_d, map_location=device))
model.to(device)
model.eval()
# Визуализация предсказаний и их запись в json файл с исходным форматом
def visualisation(image, predictions, idx_to_class, height, width, image_name):
    img = image.copy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    #Из датасета
    data = {
        "img_size": {
            "height": height,
            "width": width
        },
        "bb_objects": []
    }

    # Отрисовка предсказанных боксов и заполнение JSON
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= Score and label.item() != 0:  # Исключаем фон, его не маркируем и в json не заисываем
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            data['bb_objects'].append({
                'x1': float(x_min),
                'y1': float(y_min),
                'x2': float(x_max),
                'y2': float(y_max),
                'class': idx_to_class[label.item()]
            })
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, color='red', linewidth=2)
            )
            plt.text(
                x_min, y_min, f"{idx_to_class[label.item()]}: {score:.2f}",
                color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
            )

    # Сохранение JSON
    output_path = os.path.join(pred, f"pred_{image_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(data)
    plt.axis('off')
    plt.show()


# Итоговый инференс модели
def r_inf(images_dir, model, transform, idx_to_class):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png'))]
    print(len(image_files))

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            transformed = transform(image=img)
            img_tensor = transformed['image'].to(device)
            print(f"Обработка изображения: {image_file}")

            with torch.no_grad():
                predictions = model([img_tensor])[0]

            boxes = predictions['boxes'].cpu()

            boxes[:, [0, 2]] *= width / 800  # Масштабирование по ширине
            boxes[:, [1, 3]] *= height / 800  # Масштабирование по высоте
            predictions['boxes'] = boxes
            print(boxes)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            visualisation(img, predictions, idx_to_class, height, width, image_name)

        except:
            print("Err")

# Основной код инференса
if __name__ == "__main__":


    # Запуск инференса
    r_inf(Img_d, model, inf_t, i_c)
    print("Finish")