# Импорт библиотек
import os
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# По возможности используем cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Задание констант (модель, классы, директории)
BACKBONE = 'mobilenet_v2'
MODEL_PATH = r'path_to_model'
IMAGE_FOLDER = r'path_to_img_dir'
MASK_DIR = r'path_to_pred'  # Директория для сохранения масок в градациях серого

# Инициализация модели U-Net
model = smp.Unet(BACKBONE, encoder_weights='imagenet', classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
model.to(device)

# Цвета для визуализации классов
class_colors = {
    0: (0, 0, 0),  # Черный (фон)
    1: (255, 0, 0),  # Красный (класс 1)
    2: (0, 255, 0),  # Зеленый (класс 2)
    3: (0, 0, 255)  # Синий (класс 3)
}

# Значения в градациях серого для каждого класса
gray_values = {
    0: 0,  # Фон - черный
    1: 85,  # Класс 1 - темно-серый
    2: 170,  # Класс 2 - средне-серый
    3: 255  # Класс 3 - белый
}


# Наложение маски на изображение для визуализации
def overlay_mask(image, mask):
    h, w, _ = image.shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Создаем цветную маску
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(4):  # Для каждого класса
        colored_mask[mask_resized == c] = class_colors[c]

    overlayed = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlayed


# Сохранение единой маски в градациях серого
def save_grayscale_mask(mask, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Создаем маску в градациях серого
    grayscale_mask = np.zeros_like(mask, dtype=np.uint8)
    for c in range(4):  # Для каждого класса
        grayscale_mask[mask == c] = gray_values[c]

    # Сохранение маски
    mask_path = os.path.join(output_dir, f"{filename}_grayscale_mask.png")
    cv2.imwrite(mask_path, grayscale_mask)
    print(f"Mask_saved: {mask_path}")


# Предобработка изображений
test_transform = A.Compose([
    A.Resize(640, 640), 
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Основной цикл инференса
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Проверяем расширение файла
        image_path = os.path.join(IMAGE_FOLDER, filename)
        try:

            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Уменьшаем изображение в два раза перед обработкой
            height, width, _ = frame.shape
            frame_resized = cv2.resize(frame, (width // 2, height // 2))

            transformed = test_transform(image=frame_resized)
            image_tensor = transformed["image"].unsqueeze(0).to(device)

            # Инференс модели
            with torch.no_grad():
                output = model(image_tensor)
                output = torch.softmax(output, dim=1)  # Применяем softmax для получения вероятностей классов
                predicted_mask = torch.argmax(output, dim=1).squeeze(
                    0).cpu().numpy()  # Получаем индекс класса с максимальной вероятностью

            # Наложение маски на изображение
            overlayed_frame = overlay_mask(frame_resized, predicted_mask)


            save_grayscale_mask(predicted_mask, os.path.splitext(filename)[0], MASK_DIR)

            # Визуализация результата
            cv2.imshow(f'Image: {filename}', cv2.cvtColor(overlayed_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(0) & 0xFF == ord('q'):  # Нажмите 'q' для перехода к следующему изображению
                continue  

        except Exception as e:
            print(f"Err {filename}: {e}")

# Закрытие окон
cv2.destroyAllWindows()