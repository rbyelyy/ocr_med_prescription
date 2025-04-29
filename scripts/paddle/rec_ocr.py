"""
TRAIN:
source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/rec_ocr.py \
  --mode train \
  --use_gpu \
  --pretrained_model /home/ubuntu/OCR/model/cyrillic_PP-OCRv3_rec_train/best_accuracy \
  --paddleocr_path /home/ubuntu/OCR/PaddleOCR \
  --config_path /home/ubuntu/OCR/PaddleOCR/configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml


EXPORT:
source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/rec_ocr.py \
  --mode export \
  --use_gpu \
  --trained_model_path /home/ubuntu/OCR/output/v3_cyrillic_mobile/latest \
  --export_path /home/ubuntu/OCR/inference/trained_cyrillic_PP-OCRv3_rec \
  --paddleocr_path /home/ubuntu/OCR/PaddleOCR \
  --config_path /home/ubuntu/OCR/PaddleOCR/configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml

TEST:
source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/rec_ocr.py \
  --mode infer \
  --use_gpu \
  --image_path /home/ubuntu/OCR/data/rec/train/images/ИХА_отрицательно_russia_normal_6_42.jpg \
  --model_path /home/ubuntu/OCR/inference/trained_cyrillic_PP-OCRv3_rec \
  --dict_path /home/ubuntu/OCR/PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt
"""

import argparse
import os
import torch
import paddle
from paddleocr import PaddleOCR


def check_gpu_status():
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}, Reserved: {reserved} GB, Total: {max_memory} GB")
    else:
        print("CUDA not available. Using CPU.")


def check_paddle():
    print(f"Paddle version: {paddle.__version__}")
    print(f"CUDA enabled: {paddle.is_compiled_with_cuda()}")
    paddle.utils.run_check()


def run_inference(args):
    ocr = PaddleOCR(
        use_gpu=args.use_gpu,
        rec_model_dir=args.model_path,
        rec_char_dict_path=args.dict_path,
        lang="ru",
    )

    if os.path.isdir(args.image_path):
        for img_name in os.listdir(args.image_path):
            full_path = os.path.join(args.image_path, img_name)
            result = ocr.ocr(full_path, cls=False, det=False)
            print(f"{img_name}: {result}")
    else:
        result = ocr.ocr(args.image_path, cls=False, det=False)
        print(result)


def train_model(args):
    train_cmd = f"""
    python {args.paddleocr_path}/tools/train.py \
        -c {args.config_path} \
        -o Global.pretrained_model={args.pretrained_model} \
        -o Global.use_tensorboard=true \
        -o Global.save_epoch_step=1 \
        -o Global.eval_batch_step="[0,100]"
    """
    os.system(train_cmd.strip())


def export_model(args):
    export_cmd = f"""
    python {args.paddleocr_path}/tools/export_model.py \
        -c {args.config_path} \
        -o Global.pretrained_model={args.trained_model_path} \
        -o Global.save_inference_dir={args.export_path}
    """
    os.system(export_cmd.strip())


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR Cyrillic Model Utility")
    parser.add_argument("--mode", choices=["train", "infer", "export"], required=True)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument(
        "--image_path", type=str, help="Path to image or directory for inference"
    )
    parser.add_argument(
        "--model_path", type=str, default="./inference/cyrillic_PP-OCRv3_rec"
    )
    parser.add_argument(
        "--dict_path", type=str, default="./ppocr/utils/dict/cyrillic_dict.txt"
    )
    parser.add_argument(
        "--pretrained_model", type=str, help="Path to pretrained model for training"
    )
    parser.add_argument(
        "--trained_model_path", type=str, help="Path to trained model for export"
    )
    parser.add_argument(
        "--export_path", type=str, help="Export directory for inference model"
    )
    parser.add_argument(
        "--paddleocr_path", type=str, default="/home/ubuntu/OCR/PaddleOCR"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/ubuntu/OCR/PaddleOCR/configs/rec/PP-OCRv3/multi_language/cyrillic_PP-OCRv3_rec.yml",
    )

    args = parser.parse_args()

    check_gpu_status()
    check_paddle()

    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "export":
        export_model(args)
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()
