#Рерайтер для читаемости
# Импорт необходимых библиотек
import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import retinanet_resnet50_fpn
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Директории и параметры
PRED_DIR = 'path_to_predictions'  # Папка для сохранения результатов предсказаний
MODEL_PATH = 'path_to_model'      # Путь к модели
IMAGES_DIR = 'path_to_images'     # Путь к директории с изображениями
DATASET_MEAN = [0.36149433, 0.36072833, 0.37245706]  # Среднее значение для нормализации
DATASET_STD = [0.17737652, 0.17860805, 0.20081734]   # Стандартное отклонение для нормализации
SCORE_THRESHOLD = 0.2  # Порог уверенности для предсказаний

# Классы объектов в датасете
CLASSES = ['Car', 'FacingSwitchL', 'FacingSwitchNV', 'FacingSwitchR', 'Human',
           'SignalE', 'SignalF', 'TrailingSwitchL', 'TrailingSwitchNV',
           'TrailingSwitchR', 'Wagon']

# Соответствие индекса и имени класса
IDX_TO_CLASS = {
    0: 'background', 1: 'Car', 2: 'FacingSwitchL', 3: 'FacingSwitchNV',
    4: 'FacingSwitchR', 5: 'Human', 6: 'SignalE', 7: 'SignalF',
    8: 'TrailingSwitchL', 9: 'TrailingSwitchNV', 10: 'TrailingSwitchR',
    11: 'Wagon'
}
NUM_CLASSES = len(CLASSES) + 1  # Всего классов (включая фон)

# Использование GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Преобразования для предсказания
inference_transform = A.Compose([
    A.Resize(600, 600),  # Приведение всех изображений к одному размеру
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),  # Нормализация с учетом статистики датасета
    ToTensorV2(),  # Конвертация в тензор
])

# Создаем директорию для сохранения предсказаний, если ее нет
os.makedirs(PRED_DIR, exist_ok=True)

# Загрузка модели RetinaNet с предобученным фоном
def load_model(model_path):
    model = retinanet_resnet50_fpn(
        pretrained=False,
        num_classes=NUM_CLASSES,
        pretrained_backbone=True,
        nms_thresh=0.2  # Порог для NMS
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Переводим модель в режим инференса
    return model

# Функция для визуализации предсказаний и записи их в JSON
def visualize_predictions(image, predictions, idx_to_class, height, width, image_name):
    img = image.copy()  # Копируем исходное изображение для отрисовки
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # Данные для сохранения в формате JSON
    data = {
        "img_size": {"height": height, "width": width},
        "bb_objects": []  # Список объектов на изображении
    }

    # Прорисовываем предсказания (области и метки)
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= SCORE_THRESHOLD and label.item() != 0:  # Игнорируем фон (0 класс)
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

    # Сохраняем данные в JSON
    output_path = os.path.join(PRED_DIR, f"pred_{image_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON сохранён: {output_path}")
    print(data)
    plt.axis('off')
    plt.show()

# Функция предобработки изображения для инференса
def process_image(image_path, model, transform):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, None, None, None, None  # Возвращаем None в случае ошибки
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем BGR в RGB
    height, width = img.shape[:2]

    # Применяем преобразования
    transformed = transform(image=img)
    img_tensor = transformed['image'].to(device)

    # Предсказания с использованием модели
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Масштабируем координаты боксов обратно к размеру исходного изображения
    boxes = predictions['boxes'].cpu()
    boxes[:, [0, 2]] *= width / 600
    boxes[:, [1, 3]] *= height / 600
    predictions['boxes'] = boxes

    # Имя файла без расширения
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return img, predictions, height, width, image_name

# Основная функция инференса
def run_inference(images_dir, model, transform, idx_to_class):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png'))]
    print(f"Найдено {len(image_files)} изображений для инференса")

    # Обрабатываем изображения по очереди
    for image_file in tqdm(image_files, desc="Обработка изображений"):
        image_path = os.path.join(images_dir, image_file)
        try:
            img, predictions, height, width, image_name = process_image(image_path, model, transform)
            if img is None:
                continue  # Пропускаем, если изображение не загрузилось

            # Визуализируем и сохраняем предсказания
            visualize_predictions(img, predictions, idx_to_class, height, width, image_name)

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")

if __name__ == "__main__":
    # Загружаем модель
    model = load_model(MODEL_PATH)
    # Запускаем инференс на изображениях
    run_inference(IMAGES_DIR, model, inference_transform, IDX_TO_CLASS)
    print("Инференс завершён")