# Импорт необходимых библиотек
from transformers import TableTransformerForObjectDetection
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
import os
import json  # Добавлен импорт для работы с JSON

# Определение устройства (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Загрузка предобученной модели распознавания структуры таблиц
model_name = "microsoft/table-structure-recognition-v1.1-all"
print(f"Загрузка модели: {model_name}")
structure_model = TableTransformerForObjectDetection.from_pretrained(model_name).to(
    device
)
print("Модель успешно загружена.")


# Определение класса для изменения размера изображения с сохранением пропорций
class MaxResize:
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        return resized_image


def define_detection_transforms(max_size=800):
    """Определяет трансформации для детекции объектов."""
    detection_transform = transforms.Compose(
        [
            MaxResize(max_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return detection_transform


def define_structure_transforms(max_size=1000):
    """Определяет трансформации для распознавания структуры."""
    structure_transform = transforms.Compose(
        [
            MaxResize(max_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return structure_transform


def load_cropped_table_from_path(image_path):
    """Загружает обрезанное изображение таблицы из указанного пути."""
    print(f"Загрузка обрезанного изображения таблицы: {image_path}")
    try:
        cropped_table = Image.open(image_path).convert("RGB")
        width, height = cropped_table.size
        print(f"Размеры обрезанного изображения: {width}x{height}")
        return cropped_table
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
        exit()


def check_and_create_directory(directory_path):
    """Проверяет существование директории и создает ее, если она отсутствует."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Директория {directory_path} создана.")
    return directory_path


def save_recognition_json(data, output_path):
    """Сохраняет данные распознавания в JSON-файл."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Информация распознавания сохранена в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении JSON: {e}")


def save_visualization_image(image, output_path):
    """Сохраняет визуализированное изображение."""
    try:
        image.save(output_path)
        print(f"Визуализированное изображение сохранено в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")


def perform_structure_recognition(image, detection_transform, structure_model, device):
    """Выполняет подготовку изображения и прямой проход через модель распознавания структуры."""
    # Подготовка изображения для модели
    pixel_values = detection_transform(image).unsqueeze(0).to(device)
    print(f"Размер тензора входного изображения: {pixel_values.shape}")

    # Прямой проход через модель для получения предсказаний
    print("Выполнение прямого прохода через модель...")
    with torch.no_grad():
        outputs = structure_model(pixel_values)
    print("Предсказания получены.")

    # Обновление соответствия ID и меток классов (добавление "no object")
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    return outputs, structure_id2label


# Определение трансформаций
detection_transform = define_detection_transforms()
structure_transform = define_structure_transforms()

# Загрузка обрезанного изображения таблицы
image_path = "/home/ubuntu/OCR/output/table/cropped_tables/table_0_cropped.png"
cropped_table = load_cropped_table_from_path(image_path)

# Выполнение распознавания структуры таблицы
outputs, structure_id2label = perform_structure_recognition(
    cropped_table, detection_transform, structure_model, device
)


# Функции для постобработки результатов
def box_cxcywh_to_xyxy(x):
    """Преобразует bounding boxes из формата (центр x, центр y, ширина, высота) в (верхний левый x, верхний левый y, нижний правый x, нижний правый y)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """Масштабирует bounding boxes к исходному размеру изображения."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    """Преобразует выходные данные модели в список объектов с метками, уверенностью и bounding boxes."""
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )
    return objects


# Получение обнаруженных ячеек таблицы
cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
print(f"\nОбнаружено объектов: {len(cells)}")

# Определение пути для сохранения JSON
output_dir_json = "/home/ubuntu/OCR/output/table/rec_tables"
output_path_json = os.path.join(output_dir_json, "table_0_recognition.json")

# Проверка и создание директории для JSON
check_and_create_directory(output_dir_json)

# Сохранение информации о ячейках в JSON-файл
save_recognition_json(cells, output_path_json)

# Визуализация обнаруженных ячеек на изображении
cropped_table_visualized = cropped_table.copy()
draw = ImageDraw.Draw(cropped_table_visualized)

for cell in cells:
    draw.rectangle(cell["bbox"], outline="red", width=2)

# Определение пути для сохранения визуализированного изображения
output_dir_image = "/home/ubuntu/OCR/output/table/rec_tables"
output_path_image = os.path.join(output_dir_image, "table_0_recognition.png")

# Проверка и создание директории для изображения
check_and_create_directory(output_dir_image)

# Сохранение визуализированного изображения
save_visualization_image(cropped_table_visualized, output_path_image)
