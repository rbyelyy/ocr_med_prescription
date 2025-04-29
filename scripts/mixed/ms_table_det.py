"""
python /home/ubuntu/OCR/ms_table_det.py /home/ubuntu/OCR/data/det/test/5.jpg /home/ubuntu/OCR/output/table/vizual_table.png     --padding 0.001     --min_padding 5     --output_tables_dir /home/ubuntu/OCR/output/table/cropped_tables
"""

import os
import argparse
import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
import json
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaxResize(object):
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, image):
        w, h = image.size
        if max(w, h) > self.max_size:
            if w > h:
                ratio = self.max_size / float(w)
                new_w = self.max_size
                new_h = int(ratio * float(h))
            else:
                ratio = self.max_size / float(h)
                new_h = self.max_size
                new_w = int(ratio * float(w))
            image = image.resize((new_w, new_h), Image.ANTIALIAS)
        return image


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )
    return objects


def calculate_padding(bbox, img_size, padding_percent, min_padding=5):
    """Calculate padding with consistent parameters"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    w_padding = max(min_padding, int(padding_percent * width))
    h_padding = max(min_padding, int(padding_percent * height))
    return w_padding, h_padding


def visualize_detected_tables(
    img, det_tables, out_path=None, padding_percent=0.002, min_padding=5
):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        original_bbox = det_table["bbox"]
        w_padding, h_padding = calculate_padding(
            original_bbox, img.size, padding_percent, min_padding
        )

        extended_bbox = [
            max(0, original_bbox[0] - w_padding),
            max(0, original_bbox[1] - h_padding),
            min(img.width, original_bbox[2] + w_padding),
            min(img.height, original_bbox[3] + h_padding),
        ]

        if det_table["label"] == "table":
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        elif det_table["label"] == "table rotated":
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        else:
            continue

        rect = patches.Rectangle(
            original_bbox[:2],
            original_bbox[2] - original_bbox[0],
            original_bbox[3] - original_bbox[1],
            linewidth=1,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(rect)

        rect = patches.Rectangle(
            extended_bbox[:2],
            extended_bbox[2] - extended_bbox[0],
            extended_bbox[3] - extended_bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            extended_bbox[:2],
            extended_bbox[2] - extended_bbox[0],
            extended_bbox[3] - extended_bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            extended_bbox[:2],
            extended_bbox[2] - extended_bbox[0],
            extended_bbox[3] - extended_bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor="none",
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(
        handles=[
            Patch(
                facecolor=(1, 0, 0.45),
                edgecolor=(1, 0, 0.45),
                label="Table (extended)",
                hatch="//////",
                alpha=0.3,
            ),
            Patch(
                facecolor="none",
                edgecolor=(1, 0, 0.45),
                label="Table (original)",
                linestyle="--",
                alpha=0.5,
            ),
            Patch(
                facecolor=(0.95, 0.6, 0.1),
                edgecolor=(0.95, 0.6, 0.1),
                label="Table rotated (extended)",
                hatch="//////",
                alpha=0.3,
            ),
        ],
        bbox_to_anchor=(0.5, -0.02),
        loc="upper center",
        borderaxespad=0,
        fontsize=10,
        ncol=3,
    )

    plt.gcf().set_size_inches(10, 10)
    plt.axis("off")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig


def save_table_coordinates(table_index, original_bbox, output_dir):
    """Save the coordinates of a detected table to a JSON file."""
    coordinates = {"original_bbox": [round(x, 4) for x in original_bbox]}
    base_name = f"table_{table_index}_cropped"
    json_output_path = os.path.join(output_dir, f"{base_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump(coordinates, f, indent=4)
    print(f"Saved coordinates for table {table_index} to {json_output_path}")


def crop_and_save_tables(
    img, objects, output_dir, padding_percent=0.002, min_padding=5
):
    """Crop detected tables and save them as separate images and their coordinates."""
    os.makedirs(output_dir, exist_ok=True)
    cropped_tables = []

    for i, obj in enumerate(objects):
        if obj["label"] not in ["table", "table rotated"]:
            continue

        bbox = obj["bbox"]
        w_padding, h_padding = calculate_padding(
            bbox, img.size, padding_percent, min_padding
        )

        extended_bbox = [
            max(0, bbox[0] - w_padding),
            max(0, bbox[1] - h_padding),
            min(img.width, bbox[2] + w_padding),
            min(img.height, bbox[3] + h_padding),
        ]

        cropped_img = img.crop(extended_bbox)

        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)

        # Save cropped table
        base_name = f"table_{i}_cropped"
        image_output_path = os.path.join(output_dir, f"{base_name}.png")
        cropped_img.save(image_output_path)

        # Save table coordinates
        save_table_coordinates(i, obj["bbox"], output_dir)

        cropped_tables.append(
            {
                "image": cropped_img,
                "image_path": image_output_path,
                "json_path": os.path.join(output_dir, f"{base_name}.json"),
                "original_bbox": bbox,
                "extended_bbox": extended_bbox,
                "is_rotated": obj["label"] == "table rotated",
            }
        )

    return cropped_tables


def main(
    image_path,
    output_path,
    padding_percent=0.002,
    min_padding=5,
    output_tables_dir=None,
):
    # Load model
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    model.config.id2label[len(model.config.id2label)] = "no object"
    model.to(device)

    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_resized = image.resize((int(0.6 * width), int(0.6 * height)))

    # Preprocessing
    detection_transform = transforms.Compose(
        [
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pixel_values = detection_transform(image_resized).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(pixel_values)

    objects = outputs_to_objects(outputs, image_resized.size, model.config.id2label)

    # Visualize
    fig = visualize_detected_tables(
        image_resized,
        objects,
        out_path=output_path,
        padding_percent=padding_percent,
        min_padding=min_padding,
    )

    # Crop and save tables if output directory specified
    if output_tables_dir:
        cropped_tables = crop_and_save_tables(
            image_resized,
            objects,
            output_tables_dir,
            padding_percent=padding_percent,
            min_padding=min_padding,
        )
        print(f"Saved {len(cropped_tables)} tables to {output_tables_dir}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table detection and cropping script")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "output_path", help="Path to save the output visualization image"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.002,
        help="Padding percentage around detected tables (default: 0.002)",
    )
    parser.add_argument(
        "--min_padding",
        type=int,
        default=5,
        help="Minimum padding in pixels (default: 5)",
    )
    parser.add_argument(
        "--output_tables_dir",
        type=str,
        default=None,
        help="Directory to save cropped tables (if not specified, tables won't be saved)",
    )

    args = parser.parse_args()
    main(
        args.image_path,
        args.output_path,
        padding_percent=args.padding,
        min_padding=args.min_padding,
        output_tables_dir=args.output_tables_dir,
    )
