"""
source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/det_ocr.py   --image_path /home/ubuntu/OCR/data/det/test/3.png   --det_model_dir /home/ubuntu/OCR/model/Multilingual_PP-OCRv3_det_infer   --cls_model_dir /home/ubuntu/OCR/model/ch_ppocr_mobile_v2.0_cls_infer   --lang ru   --use_angle_cls   --use_gpu   --output_image /home/ubuntu/OCR/output/det_pp_infer/3_out.png   --output_dir /home/ubuntu/OCR/output/det_pp_infer/
"""

import argparse
import torch
import paddle
import paddleocr
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os


def check_environment():
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(
            f"GPU: {gpu_stats.name}. Max memory = {max_memory} GB. Reserved = {reserved} GB."
        )
    else:
        print("CUDA not available. Using CPU.")

    print(f"Paddle version: {paddle.__version__}")
    print(f"CUDA enabled: {paddle.is_compiled_with_cuda()}")
    paddle.utils.run_check()
    print(f"PaddleOCR version: {paddleocr.__version__}")


def initialize_paddleocr(det_model_dir, cls_model_dir, use_angle_cls, lang, use_gpu):
    return paddleocr.PaddleOCR(
        use_gpu=use_gpu,
        det_model_dir=det_model_dir,
        cls_model_dir=cls_model_dir,
        use_angle_cls=use_angle_cls,
        lang=lang,
    )


def text_detection(image_path, ocr):
    result = ocr.ocr(image_path, cls=True, det=True, rec=False)
    return result


def draw_bounding_boxes(image_path, result):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    boxes = result[0]
    for box in boxes:
        box = np.array(box, dtype=np.int32)
        box = box.reshape((-1, 1, 2))
        cv2.polylines(image_np, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    return image_np


def display_image(image_np):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def save_image(image_np, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(image_np)
    img.save(output_path)
    print(f"Saved result to {output_path}")


def save_cropped_words(image_path, result, output_dir, target_size=(320, 48)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    crop_dir = os.path.join(output_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    coords_path = os.path.join(output_dir, "coords.txt")
    image = cv2.imread(image_path)

    with open(coords_path, "w") as f:
        for idx, line in enumerate(result[0]):
            box = np.array(line, dtype=np.float32)
            rect = cv2.boundingRect(box)
            x, y, w, h = rect
            cropped = image[y : y + h, x : x + w]

            # Ресайз до 320x48
            resized = cv2.resize(cropped, target_size)

            crop_name = f"word_{idx+1}.png"
            crop_path = os.path.join(crop_dir, crop_name)
            cv2.imwrite(crop_path, resized)

            # Сохраняем координаты (исходные, до ресайза)
            coords = ",".join([f"{int(pt[0])}:{int(pt[1])}" for pt in box])
            f.write(f"{crop_name}\t{coords}\n")

    print(f"[INFO] Saved {idx+1} crops and coordinates to {crop_dir} and {coords_path}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR Cyrillic Text Detection")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--det_model_dir",
        type=str,
        required=True,
        help="Path to detection model directory",
    )
    parser.add_argument(
        "--cls_model_dir",
        type=str,
        required=True,
        help="Path to classification model directory",
    )
    parser.add_argument(
        "--lang", type=str, default="ru", help="Language (e.g., 'ru', 'ch', 'en')"
    )
    parser.add_argument(
        "--use_angle_cls", action="store_true", help="Enable angle classification"
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument(
        "--output_image", type=str, help="Optional path to save the output image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save crops and coordinates",
    )

    args = parser.parse_args()

    check_environment()

    ocr = initialize_paddleocr(
        det_model_dir=args.det_model_dir,
        cls_model_dir=args.cls_model_dir,
        use_angle_cls=args.use_angle_cls,
        lang=args.lang,
        use_gpu=args.use_gpu,
    )

    result = text_detection(args.image_path, ocr)
    save_cropped_words(args.image_path, result, args.output_dir)
    image_np = draw_bounding_boxes(args.image_path, result)

    if args.output_image:
        save_image(image_np, args.output_image)
    else:
        display_image(image_np)


if __name__ == "__main__":
    main()
