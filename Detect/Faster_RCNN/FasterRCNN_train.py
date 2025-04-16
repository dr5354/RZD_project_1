# Импорт библиотек
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
import logging

# Настройка логирования
PREDICTIONS_DIR = r'path_to_predictions'
ANNOTATIONS_DIR = r"path_to_annotations"
IMAGES_DIR = r"path_to_images"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
LOG_FILE = os.path.join(PREDICTIONS_DIR, 'training.log')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# По возможности используем cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")
logging.info(f"Используемое устройство: {device}")

# Предобработка аннотаций
def load_and_preprocess_data(annotations_dir, image_dir):
    annotation_records = []
    unique_classes = set()
    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.json'):
            continue
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, f"{base_name}")
        if not os.path.exists(image_path):
            logging.warning(f"Изображение для аннотации {filename} не найдено")
            continue
        with open(os.path.join(annotations_dir, filename), 'r') as f:
            annotation_data = json.load(f)
        boxes, labels = [], []
        img = Image.open(image_path)
        img_width, img_height = img.size
        for obj in annotation_data.get('bb_objects', []):
            class_name = obj.get('class')
            if not class_name:
                continue
            x_min, y_min, x_max, y_max = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            if x_min >= x_max or y_min >= y_max:
                logging.warning(
                    f"Некорректный bounding box в {filename}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                continue
            if x_max > img_width or y_max > img_height or x_min < 0 or y_min < 0:
                logging.warning(
                    f"Bounding box вне границ изображения в {filename}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                continue

            # Оставляем координаты в абсолютных значениях (убрана нормализация для совместимости с Faster R-CNN)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_name)
            unique_classes.add(class_name)

        if boxes:
            annotation_records.append({
                'image_path': image_path,
                'boxes': np.array(boxes, dtype=np.float32),
                'labels_str': labels
            })
        else:
            logging.warning(f"Изображение {image_path} не содержит валидных bounding boxes")

    logging.info(f"Классы: {sorted(unique_classes)}")
    return annotation_records, sorted(unique_classes)

# Класс датасета
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, class_to_idx, transform=None):
        self.annotations = annotations
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['image_path']
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Не удалось загрузить изображение: {img_path}")
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self.annotations[idx]['boxes'].tolist()  # Уже в абсолютных значениях
        labels_str = self.annotations[idx]['labels_str']
        labels = [self.class_to_idx[label] for label in labels_str]

        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
            boxes = [list(box) for box in boxes if (box[2] - box[0] > 0 and box[3] - box[1] > 0)]

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long)

        if len(boxes) == 0:
            logging.warning(f"Изображение {img_path} не содержит валидных bounding boxes после трансформации")

        return img, boxes, labels

# Обработка пакета обучающих данных
def collate_fn(batch):
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]
    return images, targets

# Визуализация предобработанных изображений и боксов
def visualize_preprocessed_images(dataset, idx_to_class, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for i in range(num_samples):
        img, boxes, labels = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            axes[i].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red'))
            axes[i].text(x_min, y_min, idx_to_class[label.item()], color='red', fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.5))
        axes[i].axis('off')
    plt.savefig(os.path.join(PREDICTIONS_DIR, "preprocessed_samples.png"))
    plt.close()

# Визуализация предсказаний
def visualize_predictions(images, preds, targets, idx_to_class, epoch, batch_idx):
    img = images[0].cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # gt боксы (зеленые)
    for box, label in zip(targets[0]['boxes'], targets[0]['labels']):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        plt.gca().add_patch(
            plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_max, fill=False, color='green', linewidth=2))
        plt.text(x_min, y_min, idx_to_class[label.item()], color='green', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))

    # pred боксы (красные)
    for box, label, score in zip(preds[0]['boxes'], preds[0]['labels'], preds[0]['scores']):
        if score > 0.1:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_max, fill=False, color='red', linewidth=2))
            plt.text(x_min, y_min, f"{idx_to_class[label.item()]}: {score:.2f}", color='red', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"Эпоха {epoch}, Батч {batch_idx}")
    plt.savefig(os.path.join(PREDICTIONS_DIR, f"epoch_{epoch}_batch_{batch_idx}.png"))
    plt.close()

# Взвешенная функция потерь для учета дисбаланса классов
def weighted_loss(loss_dict, weights):
    loss_classifier = loss_dict['loss_classifier'] * weights.mean()
    loss_box_reg = loss_dict['loss_box_reg']
    loss_objectness = loss_dict['loss_objectness'] * 0.5
    loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
    total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

    if torch.isnan(total_loss):
        logging.error(f"Итоговая потеря стала nan: {loss_dict}")
        return torch.tensor(0.0, requires_grad=True).to(device)

    return total_loss

# Основной код обучения
if __name__ == "__main__":
    # Загрузка данных с учетом фона (0 значение)
    annotations, unique_classes = load_and_preprocess_data(ANNOTATIONS_DIR, IMAGES_DIR)
    class_to_idx = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(unique_classes) + 1
    logging.info(f"Количество классов: {num_classes}")

    # Вычисленные mean и std
    DATASET_MEAN, DATASET_STD = [0.36149433, 0.36072833, 0.37245706], [0.17737652, 0.17860805, 0.20081734]
    logging.info(f"Вычисленные mean: {DATASET_MEAN}, std: {DATASET_STD}")

    '''Трансформации, отказались от агрессивной аугментации, так как она негативно влияла на процесс обучения
       (ну и задача достаточно специфичная для применения и геометрических и визуальных трансформаций)'''
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.05),
        A.RandomBrightnessContrast(brightness_limit=0.001, contrast_limit=0.001, p=0.03),
        A.Rotate(limit=2, p=0.01),
        A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=0, p=0.03),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0))

    val_transform = A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0))

    # Разделение данных 80% обучение 20% валидация
    train_size = int(0.8 * len(annotations))
    val_size = len(annotations) - train_size
    train_annotations, val_annotations = torch.utils.data.random_split(annotations, [train_size, val_size])

    train_dataset = CustomDataset(train_annotations, class_to_idx, transform=train_transform)
    val_dataset = CustomDataset(val_annotations, class_to_idx, transform=val_transform)

    # Визуализация предобработанных изображений
    visualize_preprocessed_images(train_dataset, idx_to_class, num_samples=5)

    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn,
                            num_workers=4)

    # Инициализация модели Faster R-CNN
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.IMAGENET1K_V2  # Backbone - предобученный ImageNet
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        nms_thresh=0.5,
        score_thresh=0.1
    )
    model.to(device)

    # Решение классового дисбаланса при обучении
    class_counts = {cls: 0 for cls in unique_classes}
    for annotation in annotations:
        for label in annotation['labels_str']:
            class_counts[label] += 1
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / (len(unique_classes) * max(count, 1)) for cls, count in class_counts.items()}
    weights = [class_weights[idx_to_class[i]] for i in range(1, num_classes)]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logging.info(f"Веса классов: {weights.tolist()}")

    # Адаптивный оптимизатор и scheduler
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': [p for n, p in model.named_parameters() if "backbone" not in n], 'lr': 0.001}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    # Цикл обучения
    num_epochs = 100
    best_map = 0.0
    patience = 10
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = weighted_loss(loss_dict, weights)
            train_loss += losses.item()
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Эпоха {epoch + 1}, Средняя потеря на тренировке: {avg_train_loss:.4f}")

        # Валидация
        model.eval()
        map_metric = MeanAveragePrecision()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = [img.to(device) for img in images]
                outputs = model(images)
                preds = [{'boxes': o['boxes'], 'scores': o['scores'], 'labels': o['labels']} for o in outputs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Визуализация предсказаний каждые 5 эпох, хранятся в директории предсказаний
                if (epoch + 1) % 5 == 0 and batch_idx == 0:
                    visualize_predictions(images, preds, targets, idx_to_class, epoch + 1, batch_idx)

                map_metric.update(preds, targets)

            map_result = map_metric.compute()
            current_map = map_result['map'].item()
            logging.info(f"Эпоха {epoch + 1}, mAP: {current_map:.4f}")

            scheduler.step(current_map)

            # Early stopping и сохранение лучшей модели
            if current_map > best_map:
                best_map = current_map
                torch.save(model.state_dict(), os.path.join(PREDICTIONS_DIR, "best_model_rcnn+resnet50(fpn).pth"))
                logging.info("Лучшая модель сохранена")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Ранняя остановка на эпохе {epoch + 1}")
                    break

    print(f"Обучение завершено. Лучший mAP: {best_map:.4f}")