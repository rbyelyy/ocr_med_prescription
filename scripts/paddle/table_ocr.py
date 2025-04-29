"""
source /home/ubuntu/ocr_venv_py38/bin/activate && python /home/ubuntu/OCR/scripts/paddle/table_ocr.py \
  --image_path /home/ubuntu/OCR/output/table/cropped_tables/table_0_cropped.png \
  --det_model_dir /home/ubuntu/OCR/model/Multilingual_PP-OCRv3_det_infer \
  --rec_model_dir /home/ubuntu/OCR/inference/trained_cyrillic_PP-OCRv3_rec \
  --table_model_dir /home/ubuntu/OCR/inference/en_ppstructure_mobile_v2.0_SLANet_infer \
  --rec_char_dict_path /home/ubuntu/OCR/PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt \
  --table_char_dict_path /home/ubuntu/OCR/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt \
  --output_dir /home/ubuntu/OCR/output/table/paddle_pipeline \
  --use_gpu 
"""

import os
import subprocess
import logging
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] ppocr %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR Table Recognition Script")

    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument(
        "--det_model_dir", required=True, help="Path to detection model"
    )
    parser.add_argument(
        "--rec_model_dir", required=True, help="Path to recognition model"
    )
    parser.add_argument(
        "--table_model_dir", required=True, help="Path to table structure model"
    )
    parser.add_argument(
        "--rec_char_dict_path",
        required=True,
        help="Path to recognition char dictionary",
    )
    parser.add_argument(
        "--table_char_dict_path",
        required=True,
        help="Path to table structure char dictionary",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "python3",
        "/home/ubuntu/OCR/PaddleOCR/ppstructure/table/predict_table.py",
        f"--det_model_dir={args.det_model_dir}",
        f"--rec_model_dir={args.rec_model_dir}",
        f"--table_model_dir={args.table_model_dir}",
        f"--rec_char_dict_path={args.rec_char_dict_path}",
        f"--table_char_dict_path={args.table_char_dict_path}",
        f"--image_dir={args.image_path}",
        f"--output={args.output_dir}",
        f"--use_gpu={'True' if args.use_gpu else 'False'}",
    ]

    logging.info("Running predict_table.py through subprocess")
    logging.info("Command: " + " ".join(cmd))

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        logging.info("Command output:\n" + result.stdout)
        if result.stderr:
            logging.warning("Command errors:\n" + result.stderr)

        # Check if Excel file was created
        excel_path = os.path.join(
            args.output_dir,
            os.path.basename(args.image_path).rsplit(".", 1)[0] + ".xlsx",
        )
        if os.path.exists(excel_path):
            logging.info(f"✅ Excel file saved at: {excel_path}")
        else:
            logging.error(f"❌ Excel file not found at expected path: {excel_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Subprocess failed with code {e.returncode}")
        logging.error(f"stderr:\n{e.stderr}")

    end_time = time.time()
    logging.info(f"⏱ Total processing time: {end_time - start_time:.3f}s")


if __name__ == "__main__":
    main()
