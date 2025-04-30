import os
import subprocess
import shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Папка для загрузок
UPLOAD_FOLDER = '/home/ubuntu/OCR/data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Путь к выходной папке
OUTPUT_FOLDER = '/home/ubuntu/OCR/output/table'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Команда /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("PaddleOCR", callback_data="paddle"),
            InlineKeyboardButton("Mixed Approach", callback_data="mix")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет! Выберите подход обработки таблиц:",
        reply_markup=reply_markup
    )

# --- Обработка выбора метода ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["ocr_method"] = query.data
    await query.edit_message_text(
        f"Выбран метод: {query.data}. Теперь отправьте мне фото с таблицей для обработки."
    )

# --- Команда /clean ---
async def clean(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            await update.message.reply_text("✅ Папка загрузок зачищена!")
        else:
            await update.message.reply_text("⚠️ Папка загрузок уже пуста.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при очистке: {str(e)}")

# --- Обработка фото ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте именно фотографию.")
        return

    ocr_method = context.user_data.get("ocr_method", None)
    if not ocr_method:
        keyboard = [
            [
                InlineKeyboardButton("PaddleOCR", callback_data="paddle"),
                InlineKeyboardButton("Mixed Approach", callback_data="mix")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Сначала выберите подход обработки таблиц:",
            reply_markup=reply_markup
        )
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_filename = f"{file.file_unique_id}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    await file.download_to_drive(image_path)
    await update.message.reply_text(f"✅ Фото сохранено. Идет обработка методом {ocr_method}...")

    if ocr_method == "paddle":
        await process_with_paddle(update, context, image_path, image_filename)
    elif ocr_method == "mix":
        await process_with_mix(update, context, image_path, image_filename)

# --- Обработка методом PaddleOCR ---
async def process_with_paddle(update: Update, context: ContextTypes.DEFAULT_TYPE, image_path, image_filename):
    original_excel_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(image_filename)[0] + ".xlsx")
    original_image_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(image_filename)[0] + ".jpg")

    renamed_excel_path = os.path.join(OUTPUT_FOLDER, "table_from_photo.xlsx")
    renamed_image_path = os.path.join(OUTPUT_FOLDER, "det_from_photo.jpg")

    command = f"bash -c 'source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/table_ocr.py \
        --image_path {image_path} \
        --det_model_dir /home/ubuntu/OCR/model/Multilingual_PP-OCRv3_det_infer \
        --rec_model_dir /home/ubuntu/OCR/inference/trained_cyrillic_PP-OCRv3_rec \
        --table_model_dir /home/ubuntu/OCR/inference/en_ppstructure_mobile_v2.0_SLANet_infer \
        --rec_char_dict_path /home/ubuntu/OCR/PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt \
        --table_char_dict_path /home/ubuntu/OCR/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt \
        --output_dir {OUTPUT_FOLDER} \
        --use_gpu'"

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            await update.message.reply_text(f"❌ Ошибка при обработке изображения:\n{result.stderr}")
            return
        await send_results(update, original_excel_path, original_image_path, renamed_excel_path, renamed_image_path)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка Paddle: {str(e)}")

# --- Обработка методом Mixed ---
async def process_with_mix(update: Update, context: ContextTypes.DEFAULT_TYPE, image_path, image_filename):
    cropped_tables_dir = os.path.join(OUTPUT_FOLDER, "cropped_tables")
    vizual_table_path = os.path.join(OUTPUT_FOLDER, "vizual_table.png")

    commands = [
        f"bash -c 'source /home/ubuntu/ms_easy_ocr/bin/activate && python /home/ubuntu/OCR/scripts/mixed/ms_table_det.py {image_path} {vizual_table_path} --padding 0.001 --min_padding 5 --output_tables_dir {cropped_tables_dir}'",
        f"bash -c 'source /home/ubuntu/ms_easy_ocr/bin/activate && python /home/ubuntu/OCR/scripts/mixed/ms_table_rec.py'",
        f"bash -c 'source /home/ubuntu/ms_easy_ocr/bin/activate && python /home/ubuntu/OCR/scripts/mixed/ms_table_text_rec.py'"
    ]

    try:
        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                await update.message.reply_text(f"❌ Ошибка команды:\n{cmd}\n\n{result.stderr}")
                return

        # Пути к файлам
        tesseract_excel = '/home/ubuntu/OCR/output/table/table_cells/table_data_tesseract.xlsx'
        cropped_img = os.path.join(OUTPUT_FOLDER, "cropped_tables/table_0_cropped.png")
        recognized_img = os.path.join(OUTPUT_FOLDER, "rec_tables/table_0_recognition.png")
        vizual_table_path = os.path.join(OUTPUT_FOLDER, "vizual_table.png")

        if os.path.exists(tesseract_excel):
            await update.message.reply_document(open(tesseract_excel, 'rb'), filename='таблица.xlsx')

            for filepath in [vizual_table_path, cropped_img, recognized_img]:
                if os.path.exists(filepath):
                    await update.message.reply_document(document=open(filepath, 'rb'))

            await update.message.reply_text("✅ Таблица успешно распознана методом Mix!")
        else:
            await update.message.reply_text("❌ Ошибка: не найден Excel файл.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка Mix: {str(e)}")

# --- Отправка результатов (Paddle) ---
async def send_results(update, original_excel_path, original_image_path, renamed_excel_path, renamed_image_path):
    if os.path.exists(original_excel_path):
        if os.path.exists(renamed_excel_path):
            os.remove(renamed_excel_path)
        os.rename(original_excel_path, renamed_excel_path)

        if os.path.exists(original_image_path):
            if os.path.exists(renamed_image_path):
                os.remove(renamed_image_path)
            os.rename(original_image_path, renamed_image_path)

            await update.message.reply_document(document=open(renamed_excel_path, 'rb'))
            await update.message.reply_document(document=open(renamed_image_path, 'rb'))
            await update.message.reply_text("✅ Таблица успешно распознана и отправлена!")
        else:
            await update.message.reply_document(document=open(renamed_excel_path, 'rb'))
            await update.message.reply_text("✅ Таблица успешно распознана (без визуализации)!")
    else:
        await update.message.reply_text("❌ Не найден результат обработки (Excel файл).")

# --- Запуск бота ---
def main():
    app = (
        ApplicationBuilder()
        .token("8196793459:AAFaVSKxLx_tIJ2cIFynwVs6fgjJGlexmRE")
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clean", clean))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
