import json
import os
from PIL import Image, ImageDraw
import numpy as np
import easyocr
from tqdm.auto import tqdm
import csv
import pandas as pd
import pytesseract

# --- Пути к файлам ---
json_file_path = "/home/ubuntu/OCR/output/table/rec_tables/table_0_recognition.json"
image_path = "/home/ubuntu/OCR/output/table/cropped_tables/table_0_cropped.png"
output_dir = "/home/ubuntu/OCR/output/table/table_cells"
coordinates_output_path = os.path.join(output_dir, "all_cell_data_tesseract.json")
excel_output_path_tesseract = os.path.join(output_dir, "table_data_tesseract.xlsx")
excel_output_path_easyocr = os.path.join(output_dir, "table_data_easyocr.xlsx")
tessdata_dir_config = '--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata"'


# --- Функции ---
def load_cells_from_json(json_path):
    try:
        with open(json_path, "r") as f:
            cells = json.load(f)
        return cells
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Ошибка: Некорректный формат JSON в файле {json_path}")
        return []
    except Exception as e:
        print(f"Произошла ошибка при загрузке JSON: {e}")
        return []


def extract_and_recognize_cells_tesseract(
    cells_data, image_path, output_dir, tessdata_config
):
    """Извлекает и распознает содержимое ячеек таблицы с помощью Tesseract."""
    if not cells_data:
        print("Нет данных о ячейках для обработки (Tesseract).")
        return None

    rows = sorted(
        [entry for entry in cells_data if entry["label"] == "table row"],
        key=lambda x: x["bbox"][1],
    )
    columns = sorted(
        [entry for entry in cells_data if entry["label"] == "table column"],
        key=lambda x: x["bbox"][0],
    )

    all_rows_text = []

    try:
        image = Image.open(image_path).convert("RGB")
        for i, row in enumerate(rows):
            row_text_data = []
            for j, col in enumerate(columns):
                cell_bbox = [
                    col["bbox"][0],
                    row["bbox"][1],
                    col["bbox"][2],
                    row["bbox"][3],
                ]
                cropped_cell = image.crop(cell_bbox)
                text = pytesseract.image_to_string(
                    cropped_cell, lang="rus", config=tessdata_config
                ).strip()
                row_text_data.append(text)
            all_rows_text.append(row_text_data)
        return all_rows_text
    except FileNotFoundError:
        print(f"Ошибка: Файл изображения не найден по пути {image_path} (Tesseract).")
        return None
    except Exception as e:
        print(f"Произошла ошибка при обработке ячеек (Tesseract): {e}")
        return None


def extract_cell_coordinates(cells_data):
    """Извлекает координаты ячеек таблицы, организуя их по строкам."""
    rows = [entry for entry in cells_data if entry["label"] == "table row"]
    columns = [entry for entry in cells_data if entry["label"] == "table column"]
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = [
                column["bbox"][0],
                row["bbox"][1],
                column["bbox"][2],
                row["bbox"][3],
            ]
            row_cells.append(cell_bbox)
        cell_coordinates.append(row_cells)
    return cell_coordinates


def apply_ocr_easyocr(image_path, cell_coordinates):
    """Применяет OCR к каждой ячейке таблицы с помощью easyOCR."""
    if not cell_coordinates:
        print("Нет координат ячеек для обработки (easyOCR).")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        reader = easyocr.Reader(["ru"])
        all_rows_text = []
        for row_coords in tqdm(cell_coordinates, desc="OCR (easyocr) processing"):
            row_text = []
            for cell_bbox in row_coords:
                cell_image = np.array(image.crop(cell_bbox))
                result = reader.readtext(
                    cell_image,
                    decoder="beamsearch",
                    beamWidth=7,
                    allowlist="абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,-/%()",
                    contrast_ths=0.2,
                    adjust_contrast=0.6,
                )
                text = " ".join([res[1] for res in result]) if result else ""
                row_text.append(text)
            all_rows_text.append(row_text)
        return all_rows_text
    except FileNotFoundError:
        print(f"Ошибка: Файл изображения не найден по пути {image_path} (easyOCR).")
        return None
    except Exception as e:
        print(f"Произошла ошибка при обработке ячеек (easyOCR): {e}")
        return None


def save_to_excel(data, output_path):
    """Сохраняет данные в Excel файл."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        print(f"\nДанные таблицы сохранены в: {output_path}")
        return True
    else:
        print(f"Нет данных для сохранения в {output_path}.")
        return False


# --- Основной Блок Выполнения ---

if __name__ == "__main__":
    # Загрузка данных о ячейках из JSON
    cells_data = load_cells_from_json(json_file_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Директория вывода создана или уже существует: {output_dir}")

    if cells_data:
        print("Данные о ячейках успешно загружены.")

        # Распознавание с помощью Tesseract
        tesseract_results = extract_and_recognize_cells_tesseract(
            cells_data, image_path, output_dir, tessdata_dir_config
        )
        if tesseract_results:
            save_to_excel(tesseract_results, excel_output_path_tesseract)

        # Извлечение координат ячеек для easyOCR
        cell_coordinates = extract_cell_coordinates(cells_data)

        # Распознавание с помощью easyOCR
        easyocr_results = apply_ocr_easyocr(image_path, cell_coordinates)
        if easyocr_results:
            save_to_excel(easyocr_results, excel_output_path_easyocr)

    else:
        print("Не удалось загрузить данные о таблице, обработка остановлена.")
