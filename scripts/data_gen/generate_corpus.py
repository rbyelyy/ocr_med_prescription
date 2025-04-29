"""
python generate_corpus.py --max_words 10 --clean --output_dir /home/ubuntu/OCR/data/rec --base_dir /home/ubuntu/OCR/data/rec --test_split 0.2
"""

import sys
import os
import re
import random
import time
import shutil
import argparse
from datetime import datetime

# Добавляем директорию TextRecognitionDataGenerator в путь
sys.path.append("/home/ubuntu/OCR/TextRecognitionDataGenerator")
from trdg.generators import GeneratorFromStrings

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(
    description="Генератор данных для OCR на русском языке"
)

parser.add_argument(
    "--max_words",
    type=int,
    default=200,
    help="Максимальное количество слов для обработки (0 = все)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/ubuntu/OCR/dataset",
    help="Директория для сохранения результатов",
)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Очистить директорию перед генерацией новых данных",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="",
    help="Базовая директория для путей в файле аннотаций (для использования в других системах)",
)
parser.add_argument(
    "--test_split",
    type=float,
    default=0.2,
    help="Доля данных для тестового набора (от 0.0 до 1.0)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed для случайных операций, чтобы обеспечить воспроизводимость результатов",
)
args = parser.parse_args()

# Устанавливаем seed для воспроизводимости
random.seed(args.seed)

text_file = "/home/ubuntu/OCR/corpus/rus_corpus.txt"

font_path1 = "/home/ubuntu/OCR/fonts/ocr_russia.ttf"
font_path2 = "/home/ubuntu/OCR/fonts/ocr_calibri.ttf"
output_dir = args.output_dir
images_dir = os.path.join(output_dir, "images")

# Создаем директории для тренировочного и тестового наборов
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
train_images_dir = os.path.join(train_dir, "images")
test_images_dir = os.path.join(test_dir, "images")

# Файлы аннотаций для тренировочного и тестового наборов
train_annotation_file = os.path.join(train_dir, "labels.txt")
test_annotation_file = os.path.join(test_dir, "labels.txt")

# Очистка директорий, если указан параметр --clean
if args.clean:
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Очистка директорий перед генерацией..."
    )

    # Удаляем файлы аннотаций, если они существуют
    for file_path in [train_annotation_file, test_annotation_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Удален файл аннотаций: {file_path}"
            )

    # Очищаем директории с изображениями, если они существуют
    for dir_path in [train_images_dir, test_images_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Очищена директория изображений: {dir_path}"
            )

# Создаем необходимые директории
for dir_path in [output_dir, train_dir, test_dir, train_images_dir, test_images_dir]:
    os.makedirs(dir_path, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Чтение корпуса из {text_file}")

# Читаем текстовый корпус
with open(text_file, "r", encoding="utf-8") as f:
    strings = [line.strip() for line in f if line.strip()]

print(
    f"[{datetime.now().strftime('%H:%M:%S')}] Прочитано {len(strings)} строк из корпуса"
)

# Ограничиваем количество слов, если корпус слишком большой и указан лимит
max_words = args.max_words
if max_words > 0 and len(strings) > max_words:
    strings = random.sample(strings, max_words)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Выбрано {max_words} случайных строк из корпуса"
    )
else:
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Обработка всех {len(strings)} строк из корпуса"
    )

# Разделение строк на тренировочный и тестовый наборы
test_size = max(0, min(1, args.test_split))  # Ограничиваем значение от 0 до 1
random.shuffle(strings)  # Перемешиваем строки для случайного разделения
split_index = int(len(strings) * (1 - test_size))
train_strings = strings[:split_index]
test_strings = strings[split_index:]

print(
    f"[{datetime.now().strftime('%H:%M:%S')}] Разделение данных: "
    f"{len(train_strings)} строк для обучения ({100 * (1-test_size):.1f}%), "
    f"{len(test_strings)} строк для тестирования ({100 * test_size:.1f}%)"
)

# Определяем настройки аугментации
augmentations = [
    {"name": "normal", "random_skew": False, "blur": 0, "distorsion_type": 0},
    {"name": "random_skew", "random_skew": True, "blur": 0, "distorsion_type": 0},
    {"name": "blur_light", "random_skew": False, "blur": 0.5, "distorsion_type": 0},
    {"name": "blur_medium", "random_skew": False, "blur": 1, "distorsion_type": 0},
    {"name": "blur_heavy", "random_skew": False, "blur": 1.5, "distorsion_type": 0},
    {"name": "distortion_sin", "random_skew": False, "blur": 0, "distorsion_type": 1},
    {"name": "distortion_cos", "random_skew": False, "blur": 0, "distorsion_type": 2},
    {
        "name": "distortion_random",
        "random_skew": False,
        "blur": 0,
        "distorsion_type": 3,
    },
    {"name": "combined1", "random_skew": True, "blur": 0.5, "distorsion_type": 1},
    {"name": "combined2", "random_skew": True, "blur": 1, "distorsion_type": 2},
]

# Настройка размеров шрифта для разного типа текста
base_font_sizes = [32, 36, 42, 48]


# Функция для адаптации размера шрифта и ширины изображения в зависимости от длины текста
def adapt_font_and_width(text):
    text_len = len(text)

    # Определяем размер шрифта в зависимости от длины
    if text_len > 30:
        # Для очень длинных текстов уменьшаем шрифт
        font_size = random.choice([24, 28, 32])
    elif text_len > 20:
        # Для длинных текстов
        font_size = random.choice([28, 32, 36])
    elif text_len > 10:
        # Для средних текстов
        font_size = random.choice([32, 36, 42])
    else:
        # Для коротких текстов
        font_size = random.choice([36, 42, 48])

    # Определяем ширину в зависимости от длины
    if text_len > 30:
        width = 720  # Очень широкое изображение
    elif text_len > 20:
        width = 600  # Широкое изображение
    elif text_len > 10:
        width = 480  # Среднее изображение
    else:
        width = 320  # Стандартное изображение

    return font_size, width


# Функция для генерации изображений для заданного набора строк и записи аннотаций
def generate_images(strings_set, images_dir, annotation_file, set_name):
    total_images = 0
    start_time = time.time()

    with open(annotation_file, "w", encoding="utf-8") as anno_file:
        # Обрабатываем каждое слово
        for word_idx, word in enumerate(strings_set):
            # Адаптируем размер шрифта и ширину изображения в зависимости от длины текста
            font_size, width = adapt_font_and_width(word)

            # Для каждого шрифта
            for font_idx, font_path in enumerate([font_path1, font_path2]):
                font_name = "russia" if font_idx == 0 else "calibri"

                # Выбираем случайные аугментации для этого слова и шрифта
                # (не все 10 для ускорения и уменьшения избыточности)
                selected_augs = random.sample(augmentations, k=3)
                selected_augs.append(
                    augmentations[0]
                )  # Всегда добавляем нормальный вариант

                # Для выбранных аугментаций
                for aug in selected_augs:
                    # Генерируем изображение для текущего слова
                    generator = GeneratorFromStrings(
                        [word],
                        count=1,
                        fonts=[font_path],
                        size=font_size,
                        width=width,  # Адаптивная ширина
                        language="ru",
                        random_skew=aug["random_skew"],
                        blur=aug["blur"],
                        distorsion_type=aug["distorsion_type"],
                        distorsion_orientation=1 if aug["distorsion_type"] > 0 else 0,
                        background_type=random.randint(0, 1),  # Случайный тип фона
                    )

                    # Обрабатываем сгенерированное изображение
                    for img, text in generator:
                        # Очищаем текст для имени файла
                        safe_text = re.sub(r"[^\w\s-]", "", text)
                        safe_text = re.sub(r"\s+", "_", safe_text)

                        if len(safe_text) > 30:
                            safe_text = safe_text[:30]

                        if not safe_text:
                            safe_text = f"image_{total_images}"

                        # Создаем имя файла
                        file_name = f"{safe_text}_{font_name}_{aug['name']}_{word_idx}_{font_size}"
                        image_path = f"{images_dir}/{file_name}.jpg"

                        # Сохраняем изображение
                        img.save(image_path)

                        # Записываем аннотацию
                        if args.base_dir:
                            # Используем указанный базовый путь
                            if args.base_dir.endswith("/"):
                                base = args.base_dir
                            else:
                                base = args.base_dir + "/"
                            rel_image_path = f"{base}{set_name}/images/{os.path.basename(image_path)}"
                        else:
                            # Используем относительный путь от набора данных
                            rel_image_path = os.path.relpath(
                                image_path, start=os.path.dirname(annotation_file)
                            )

                        anno_file.write(f"{rel_image_path}\t{text}\n")

                        total_images += 1

                        # Показываем прогресс
                        if total_images % 50 == 0:
                            elapsed_time = time.time() - start_time
                            images_per_second = (
                                total_images / elapsed_time if elapsed_time > 0 else 0
                            )
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] {set_name}: Сгенерировано {total_images} изображений... "
                                f"({images_per_second:.1f} изобр./сек)"
                            )

    # Возвращаем статистику
    elapsed_time = time.time() - start_time
    return total_images, elapsed_time


# Генерируем тренировочный набор данных
print(
    f"\n[{datetime.now().strftime('%H:%M:%S')}] Генерация тренировочного набора данных..."
)
train_images, train_time = generate_images(
    train_strings, train_images_dir, train_annotation_file, "train"
)

# Генерируем тестовый набор данных
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Генерация тестового набора данных...")
test_images, test_time = generate_images(
    test_strings, test_images_dir, test_annotation_file, "test"
)

# Общая статистика
total_images = train_images + test_images
total_time = train_time + test_time

# Печатаем итоги
print(
    f"\n[{datetime.now().strftime('%H:%M:%S')}] Генерация завершена за {total_time:.1f} сек"
)
print(f"Всего сгенерировано: {total_images} изображений")
print(f"- Тренировочный набор: {train_images} изображений ({train_time:.1f} сек)")
print(f"- Тестовый набор: {test_images} изображений ({test_time:.1f} сек)")
print(f"- Обработано {len(strings)} строк из корпуса")
print(f"- Использованы адаптивные размеры шрифта в зависимости от длины текста")
print(f"- Использованы адаптивные размеры изображений в зависимости от длины текста")
print(f"- Использованы шрифты: ocr_russia.ttf и ocr_calibri.ttf")
print(f"- Аугментации включают: обычный текст, наклон, размытие и искажения")
print(f"Структура данных:")
print(f"- Тренировочный набор: {train_dir}")
print(f"  - Изображения: {train_images_dir}")
print(f"  - Аннотации: {train_annotation_file}")
print(f"- Тестовый набор: {test_dir}")
print(f"  - Изображения: {test_images_dir}")
print(f"  - Аннотации: {test_annotation_file}")
if args.base_dir:
    print(
        f"Для путей в файле аннотаций используется базовая директория: {args.base_dir}"
    )

# Печатаем статистику по размерам текста
text_length_stats = {}
for word in strings:
    length = len(word)
    if length <= 10:
        category = "короткие (≤10)"
    elif length <= 20:
        category = "средние (11-20)"
    elif length <= 30:
        category = "длинные (21-30)"
    else:
        category = "очень длинные (>30)"

    text_length_stats[category] = text_length_stats.get(category, 0) + 1

print("\nСтатистика по длине текста:")
for category, count in sorted(text_length_stats.items()):
    percentage = (count / len(strings)) * 100
    print(f"- {category}: {count} строк ({percentage:.1f}%)")

# Подсчитываем статистику по генерации для каждого набора
print("\nСтатистика по тренировочному набору:")
train_file_count = len([f for f in os.listdir(train_images_dir) if f.endswith(".jpg")])
train_file_size = sum(
    os.path.getsize(os.path.join(train_images_dir, f))
    for f in os.listdir(train_images_dir)
    if f.endswith(".jpg")
)
train_file_size_mb = train_file_size / (1024 * 1024)
print(f"- Количество файлов: {train_file_count}")
print(f"- Общий размер данных: {train_file_size_mb:.2f} МБ")

print("\nСтатистика по тестовому набору:")
test_file_count = len([f for f in os.listdir(test_images_dir) if f.endswith(".jpg")])
test_file_size = sum(
    os.path.getsize(os.path.join(test_images_dir, f))
    for f in os.listdir(test_images_dir)
    if f.endswith(".jpg")
)
test_file_size_mb = test_file_size / (1024 * 1024)
print(f"- Количество файлов: {test_file_count}")
print(f"- Общий размер данных: {test_file_size_mb:.2f} МБ")

# Проверяем и выводим примеры данных
print("\nПримеры записей из файла аннотаций тренировочного набора:")
with open(train_annotation_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i in range(min(3, len(lines))):
        print(lines[i].strip())

print("\nПримеры записей из файла аннотаций тестового набора:")
with open(test_annotation_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i in range(min(3, len(lines))):
        print(lines[i].strip())
