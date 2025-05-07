"""
Medical Prescription OCR System

This module provides functionality to extract text from medical prescriptions
using vision-language models, with a focus on multi-language support.
"""

import json
import re
import ast
import os
import base64
import glob
import argparse
import torch
import logging
import time
import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datetime import datetime

# --- Logging Setup ---


class LoggingManager:
    """Handles logging configuration and operations."""

    @staticmethod
    def setup_logging(
        log_file: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> logging.Logger:
        """Set up logging to file and console."""
        # Create logger
        logger = logging.getLogger("OCR_System")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers if any
        if logger.handlers:
            logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Create file handler if log file is specified
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_execution_time(func):
        """Decorator to log function execution time."""

        def wrapper(*args, **kwargs):
            logger = logging.getLogger("OCR_System")
            start_time = time.time()

            logger.debug(f"Starting {func.__name__}")
            result = func(*args, **kwargs)

            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"Finished {func.__name__} in {execution_time:.2f} seconds")

            return result

        return wrapper


# --- Model Management ---


class ModelManager:
    """Handles loading and caching of vision-language models."""

    _model_cache = {}
    _logger = logging.getLogger("OCR_System.ModelManager")

    @staticmethod
    def load_model(
        checkpoint="Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="/home/ubuntu/model_cache"
    ) -> Tuple[Any, Any]:
        """Load model with persistent file caching."""
        logger = ModelManager._logger
        os.makedirs(cache_dir, exist_ok=True)

        try:
            if checkpoint in ModelManager._model_cache:
                logger.info(f"Loaded model from memory cache: {checkpoint}")
                return ModelManager._model_cache[checkpoint]

            logger.info(f"Loading model: {checkpoint}")
            start_time = time.time()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint,
                cache_dir=cache_dir,
                do_sample=True,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )

            processor = AutoProcessor.from_pretrained(
                checkpoint, cache_dir=cache_dir, local_files_only=False, use_fast=True
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")

            ModelManager._model_cache[checkpoint] = (model, processor)
            return model, processor

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return None, None

    @staticmethod
    def clean_cache():
        """Clear the model cache."""
        logger = ModelManager._logger
        logger.info("Clearing model cache")
        ModelManager._model_cache = {}

        # Also clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


# --- Image Processing ---


class ImageProcessor:
    """Handles image loading and size operations."""

    _logger = logging.getLogger("OCR_System.ImageProcessor")

    @staticmethod
    @LoggingManager.log_execution_time
    def check_file_type_and_process(
        file_path: str, output_dir: str = None
    ) -> List[str]:
        """
        Check if the file is a PDF or image and process accordingly.

        Args:
            file_path: Path to the file (PDF or image)
            output_dir: Directory to save converted images if file is PDF

        Returns:
            List of image paths (original path if image, converted paths if PDF)
        """
        logger = ImageProcessor._logger

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found at '{file_path}'")
                return []

            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()

            # If file is PDF, convert to images
            if file_ext == ".pdf":
                logger.info(f"Detected PDF file: {file_path}")
                converted_images = ImageProcessor.convert_pdf_to_images(
                    pdf_path=file_path, output_dir=output_dir
                )

                if converted_images:
                    logger.info(
                        f"Successfully converted PDF to {len(converted_images)} images"
                    )
                    return converted_images
                else:
                    logger.error("Failed to convert PDF to images")
                    return []

            # If file is an image, return its path in a list
            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"]:
                logger.info(f"Detected image file: {file_path}")
                return [file_path]

            # Unsupported file type
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return []

        except Exception as e:
            logger.error(f"Error in file type checking: {str(e)}", exc_info=True)
            return []

    @staticmethod
    @LoggingManager.log_execution_time
    def convert_pdf_to_images(
        pdf_path: str, output_dir: str = None, dpi: int = 300
    ) -> Optional[List[str]]:
        """
        Convert a PDF file to a list of images (one per page).

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images, if None uses same directory as PDF
            dpi: Resolution for the output images

        Returns:
            List of paths to the generated images, or None if conversion failed
        """
        logger = ImageProcessor._logger
        try:
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found at '{pdf_path}'")
                return None

            # If output directory not specified, use PDF directory
            if output_dir is None:
                output_dir = os.path.dirname(pdf_path)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get PDF filename without extension for naming output images
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # Convert PDF to images using pdf2image library
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=dpi)
            image_paths = []

            # Save each page as an image
            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.jpg")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)

            logger.debug(f"Converted PDF to {len(image_paths)} images at {dpi} DPI")
            return image_paths

        except ImportError:
            logger.error(
                "pdf2image library not installed. Install with: pip install pdf2image"
            )
            return None
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
            return None

    @staticmethod
    @LoggingManager.log_execution_time
    def get_image_dimensions(image_path: str) -> Tuple[int, int]:
        """Return width and height of an image."""
        logger = ImageProcessor._logger
        try:
            img = Image.open(image_path)
            width, height = img.size
            logger.debug(f"Image dimensions: {width}x{height} pixels")
            return width, height
        except FileNotFoundError:
            logger.error(f"Image not found at '{image_path}'")
            return 0, 0
        except Exception as e:
            logger.error(f"Error getting image dimensions: {str(e)}", exc_info=True)
            return 0, 0

    @staticmethod
    @LoggingManager.log_execution_time
    def load_and_resize(
        file_path: str, size: Tuple[int, int] = (600, 800), output_dir: str = None
    ) -> Optional[List[Image.Image]]:
        """
        Load a file (image or PDF) and resize it to the specified dimensions.
        If PDF, converts to images first.

        Args:
            file_path: Path to the file (PDF or image)
            size: Dimensions to resize to
            output_dir: Directory to save converted images if file is PDF

        Returns:
            List of resized PIL Image objects, or None if processing failed
        """
        logger = ImageProcessor._logger
        try:
            # Check file type and get list of image paths
            image_paths = ImageProcessor.check_file_type_and_process(
                file_path, output_dir
            )

            if not image_paths:
                logger.error(f"No valid images found for processing from {file_path}")
                return None

            # Load and resize all images
            resized_images = []
            for img_path in image_paths:
                try:
                    img = (
                        Image.open(img_path).convert("RGB").resize(size, Image.LANCZOS)
                    )
                    resized_images.append(img)
                    logger.debug(f"Image {img_path} loaded and resized to {size}")
                except Exception as e:
                    logger.error(
                        f"Error processing image {img_path}: {str(e)}", exc_info=True
                    )

            if not resized_images:
                logger.error("Failed to process any of the images")
                return None

            logger.info(f"Successfully processed {len(resized_images)} images")
            return resized_images

        except Exception as e:
            logger.error(f"Error in load_and_resize: {str(e)}", exc_info=True)
            return None

    @staticmethod
    @LoggingManager.log_execution_time
    def encode_to_base64(image_path: str) -> Optional[str]:
        """Encode an image file to a Base64 string."""
        logger = ImageProcessor._logger
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                logger.debug(f"Image encoded to base64: {len(encoded)} bytes")
                return encoded
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}", exc_info=True)
            return None


# --- OCR Inference Engine ---


class OCRInferenceEngine:
    """Handles inference operations using vision-language models."""

    _logger = logging.getLogger("OCR_System.OCRInferenceEngine")

    def __init__(self, model=None, processor=None):
        """Initialize with optional model and processor."""
        self.model = model
        self.processor = processor
        self.logger = OCRInferenceEngine._logger

    def load_model(self, checkpoint="Qwen/Qwen2.5-VL-3B-Instruct"):
        """Load the model if not already loaded."""
        if self.model is None or self.processor is None:
            self.logger.info(f"Loading model: {checkpoint}")
            self.model, self.processor = ModelManager.load_model(checkpoint)
            if self.model:
                # Compile model for faster inference if supported
                self.logger.info("Compiling model for faster inference")
                try:
                    self.model = torch.compile(self.model)
                    self.logger.info("Model compilation successful")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {str(e)}")

        status = self.model is not None and self.processor is not None
        self.logger.info(f"Model load status: {'Success' if status else 'Failed'}")
        return status

    @LoggingManager.log_execution_time
    def run_inference(
        self,
        image_path: str,
        prompt: str,
        sys_prompt: str = "You are a helpful OCR assistant for medical prescriptions.",
        max_new_tokens: int = 2048,
        return_input: bool = False,
        verbose: bool = False,
    ) -> Union[str, Tuple[str, Any]]:
        """Perform inference using the loaded vision-language model."""
        if not self.model or not self.processor:
            self.logger.error("Model or processor not available. Load model first.")
            return "" if not return_input else ("", None)

        self.logger.info(f"Running inference on image: {image_path}")

        # Load and resize image
        image = ImageProcessor.load_and_resize(image_path)
        if not image:
            self.logger.error("Failed to load image for inference")
            return "" if not return_input else ("", None)

        image_local_path = "file://" + image_path

        # Create message format expected by the model
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"image": image_local_path},
                ],
            },
        ]

        # Apply chat template and prepare inputs
        self.logger.debug("Preparing model inputs")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if verbose:
            self.logger.debug(f"Text input to model: {text}")

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate response
        self.logger.info("Generating model response")
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True
            )

        generation_time = time.time() - start_time
        self.logger.info(f"Response generated in {generation_time:.2f} seconds")

        # Decode output
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        if output_text and output_text[0]:
            self.logger.info(f"Generated response length: {len(output_text[0])} chars")
        else:
            self.logger.warning("Empty response generated")

        return (output_text[0], inputs) if return_input else output_text[0]


# --- JSON Processing ---


class JSONProcessor:
    """Handles JSON parsing and extraction from text responses."""

    _logger = logging.getLogger("OCR_System.JSONProcessor")

    @staticmethod
    @LoggingManager.log_execution_time
    def parse_json(json_output: str) -> str:
        """Parse JSON string, removing markdown fencing if present."""
        logger = JSONProcessor._logger
        logger.debug("Parsing JSON output")

        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1 :])  # Get content after ```json
                json_output = json_output.split("```")[
                    0
                ]  # Get content before closing ```
                logger.debug("Markdown fencing removed from JSON")
                break  # Stop searching after finding ```json

        return json_output

    @staticmethod
    @LoggingManager.log_execution_time
    def extract_json(
        response: str, start_marker: str = "```json"
    ) -> Optional[List[Dict]]:
        """Extract a JSON array from a string after a specified start marker."""
        logger = JSONProcessor._logger
        logger.debug(f"Extracting JSON from response (length: {len(response)})")

        json_start_index = response.find(start_marker)
        if json_start_index == -1:
            logger.warning(f"'{start_marker}' marker not found in the response")
            return None

        json_part = response[json_start_index + len(start_marker) :]
        match = re.search(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", json_part, re.DOTALL)

        if not match:
            logger.warning(
                f"No valid JSON array found after the '{start_marker}' marker"
            )
            return None

        json_str = match.group(0)
        try:
            result = json.loads(json_str)
            logger.info(f"Successfully extracted JSON array with {len(result)} items")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {str(e)}", exc_info=True)
            return None

    @staticmethod
    @LoggingManager.log_execution_time
    def save_json(data: Union[List, Dict], save_path: str) -> bool:
        """Save JSON data to a file."""
        logger = JSONProcessor._logger
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            logger.info(f"JSON data saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON data: {str(e)}", exc_info=True)
            return False


# --- Visualization ---


class OCRVisualizer:
    """Handles visualization of OCR results on images."""

    @staticmethod
    def _load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load specified font or fall back to default."""
        try:
            return ImageFont.truetype(font_name, font_size)
        except (ImportError, IOError):
            print(f"Warning: Font '{font_name}' not found. Using default font.")
            return ImageFont.load_default()

    @staticmethod
    def _find_suitable_font(font_size: int = 10) -> ImageFont.FreeTypeFont:
        """Find and load a suitable font with multi-language support."""
        font_paths = [
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for path in font_paths:
            try:
                font = ImageFont.truetype(path, size=font_size)
                print(f"Using font: {path}")
                return font
            except IOError:
                continue

        font = ImageFont.load_default()
        print(
            "Warning: Using default font. Non-Latin text might not display correctly."
        )
        return font

    @staticmethod
    def plot_bounding_boxes(
        image_path: str,
        bounding_boxes: Union[str, List[Dict]],
        width: int,
        height: int,
        save_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """Plot bounding boxes with text on an image using normalized coordinates."""
        # Load the image
        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image not found at '{image_path}'")
            return None

        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Handle JSON string input for bounding boxes
        if isinstance(bounding_boxes, str):
            bounding_boxes = JSONProcessor.parse_json(bounding_boxes)
            try:
                bounding_boxes = ast.literal_eval(bounding_boxes)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing bounding box data: {e}")
                return None

        # Find suitable font
        font = OCRVisualizer._find_suitable_font()

        # Draw each bounding box
        for i, bbox_info in enumerate(bounding_boxes):
            color = "green"
            bbox = bbox_info.get("bbox")
            text_content = bbox_info.get("text", bbox_info.get("text_content", ""))

            # Skip if bounding box info is missing
            if not bbox:
                continue

            # Convert normalized coordinates to absolute if needed
            if max(bbox) <= 1.0:  # Normalized coordinates
                abs_y1 = int(bbox[1] * height)
                abs_x1 = int(bbox[0] * width)
                abs_y2 = int(bbox[3] * height)
                abs_x2 = int(bbox[2] * width)
            else:  # Absolute coordinates
                abs_x1, abs_y1, abs_x2, abs_y2 = [int(coord) for coord in bbox]

            # Ensure x1 <= x2 and y1 <= y2
            x1, y1 = min(abs_x1, abs_x2), min(abs_y1, abs_y2)
            x2, y2 = max(abs_x1, abs_x2), max(abs_y1, abs_y2)

            # Draw the rectangle
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=1)

            # Add the text label if available
            if text_content:
                draw.text((x1, y2), text_content, fill=color, font=font)

        # Save the image if a path was provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            print(f"Image with bounding boxes saved to {save_path}")

        return img

    @staticmethod
    @LoggingManager.log_execution_time
    def render_on_canvas(
        image_path: str,
        json_response: List[Dict],
        font_size: int = 8,
        font_name: str = "arial.ttf",
        save_path: Optional[str] = None,
        width: int = 600,
        height: int = 800,
    ) -> Optional[Image.Image]:
        """Render bounding boxes and text labels on a white canvas."""
        try:
            # Load original image dimensions
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            font = OCRVisualizer._load_font(font_name, font_size)

            # Draw each annotation
            for annotation in json_response:
                text = annotation.get("text", "")
                bbox = annotation.get("bbox")

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                # Draw bounding box
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

                # Draw text label
                text_position = (x1, y1 - font_size - 2)
                draw.text(text_position, text, fill="blue", font=font)

            # Save the canvas if a path was provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                canvas.save(save_path)
                print(f"Canvas with annotations saved to {save_path}")

            return canvas

        except FileNotFoundError:
            print(f"Error: Image not found at '{image_path}'")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None


# --- Processing Functions ---


@LoggingManager.log_execution_time
def process_image(
    image_path: str,
    output_dir: str = "./output",
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> bool:
    """Process a single image and save results to output directory."""
    logger = logging.getLogger("OCR_System.processing")

    try:
        logger.info(f"Processing image: {image_path}")

        image_paths = ImageProcessor.check_file_type_and_process(image_path, output_dir)

        if not image_paths:
            logger.error(f"Failed to process input file: {image_path}")
            return None

        # Use the first converted image for dimensions and processing
        primary_image_path = image_paths[0]
        image_path = primary_image_path  # TODO need to avoid it
        dimensions = ImageProcessor.get_image_dimensions(primary_image_path)

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames based on input filename
        base_filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_filename)[0]

        json_output_path = os.path.join(output_dir, f"{name_without_ext}_results.json")
        annotated_img_path = os.path.join(
            output_dir, f"{name_without_ext}_annotated.png"
        )
        canvas_output_path = os.path.join(output_dir, f"{name_without_ext}_canvas.png")

        logger.debug(
            f"Output paths: {json_output_path}, {annotated_img_path}, {canvas_output_path}"
        )

        # OCR prompt
        prompt = """This image contains a medical prescription with text primarily in Russian.
        Analyze the image and extract all text elements. Image width: 600 pixels and height: 800 pixels.
        For each detected text element,
        identify its textual content and bounding box. Return the result as a JSON array
        of objects. Each object should have the following fields:
        - 'text': A string with the recognized text (preserving the Russian language).
        - 'bbox': An array of four integers representing the absolute coordinates of the 
          bounding box in the format [x1, y1, x2, y2], where (x1, y1) are the coordinates
          of the top-left corner, and (x2, y2) are the coordinates of the bottom-right corner.
        """

        # Get image dimensions
        logger.info("Getting image dimensions")
        width, height = ImageProcessor.get_image_dimensions(image_path)

        # Load model and run inference
        logger.info("Initializing OCR engine")
        engine = OCRInferenceEngine()
        if not engine.load_model(model_name):
            logger.error(f"Failed to load model {model_name}")
            return False

        logger.info("Running OCR inference")
        response = engine.run_inference(image_path, prompt)

        # Extract and save JSON
        logger.info("Extracting JSON from model response")
        json_response = JSONProcessor.extract_json(response)
        if not json_response:
            logger.error("Failed to extract JSON from model response")
            return False

        # Save JSON results
        logger.info("Saving results to JSON file")
        JSONProcessor.save_json(json_response, json_output_path)

        # Generate and save visualizations
        logger.info("Generating visualizations")
        OCRVisualizer.plot_bounding_boxes(
            image_path, json_response, width, height, save_path=annotated_img_path
        )

        OCRVisualizer.render_on_canvas(
            image_path,
            json_response,
            font_size=8,
            save_path=canvas_output_path,
            width=600,
            height=800,
        )

        logger.info(f"Successfully processed {image_path}")
        logger.info(f"Results saved to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        return False


def process_directory(
    input_dir: str,
    output_dir: str = "./output",
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> None:
    """Process all images in a directory and save results to output directory."""
    logger = logging.getLogger("OCR_System.processing")

    # Check if directory exists
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return

    # Get all image files
    logger.info(f"Scanning directory: {input_dir}")
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return

    total_images = len(image_files)
    logger.info(f"Found {total_images} image files to process")

    # Process each image with progress tracking
    successful = 0

    # Set up progress bar
    progress_bar = tqdm.tqdm(total=total_images, desc="Processing Images", unit="image")

    for img_path in image_files:
        progress_bar.set_description(f"Processing {os.path.basename(img_path)}")
        if process_image(img_path, output_dir, model_name):
            successful += 1
        progress_bar.update(1)

    progress_bar.close()

    logger.info(
        f"Processing complete: {successful}/{total_images} images successfully processed"
    )

    # Clean up
    logger.info("Cleaning up resources")
    ModelManager.clean_cache()


def main():
    """Example usage of the OCR system."""
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ocr_system_{timestamp}.log")

    logger = LoggingManager.setup_logging(
        log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG
    )

    logger.info("Medical Prescription OCR System starting")
    logger.info(f"Log file: {log_file}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Medical Prescription OCR System")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input image file or directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Path to output directory (default: ./output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose console output"
    )

    try:
        args = parser.parse_args()

        # Adjust logging level if verbose flag is set
        if args.verbose:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.DEBUG)
                    logger.info("Verbose logging enabled")

        # Log input parameters
        logger.info(f"Input path: {args.input}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Model: {args.model}")

        # Process input path (file or directory)
        start_time = time.time()

        if os.path.isfile(args.input):
            logger.info("Processing single image")
            process_image(args.input, args.output, args.model)
        elif os.path.isdir(args.input):
            logger.info("Processing directory of images")
            process_directory(args.input, args.output, args.model)
        else:
            logger.error(f"Input path {args.input} does not exist")

        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info("Processing complete")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
    finally:
        logger.info("OCR System shutting down")


# Example of direct script usage
if __name__ == "__main__":
    # For quick testing without command line arguments, uncomment and modify:
    # import sys
    # sys.argv = ["script.py", "--input", "/home/ubuntu/OCR/data/det/test", "--output", "./output", "--verbose"]

    main()


if __name__ == "__main__":
    main()
