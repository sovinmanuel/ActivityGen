import json
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt

from activitygen.config.parameter_config import CONFIG_PARSER
from . import file_utils as file
import os
from os.path import join as pjoin
from .processor import ImageProcessor

from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection

processor = ImageProcessor()


class TableDetector:
    def __init__(self, output_root):
        self.feature_extractor = DetrFeatureExtractor(
            do_resize=True, size=800, max_size=800
        )
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)

    def detect(self, input_img_path, resize_height=800, threshold_proba=0.7):
        image = file.read_and_resize_image(input_img_path, resize_height)
        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold_proba

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = self.feature_extractor.post_process(
            outputs, target_sizes
        )
        bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]

        return image, probas[keep], bboxes_scaled

    def plot_results_detection(self, pil_img, prob, boxes):
        plt.imshow(pil_img)
        ax = plt.gca()

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color="red",
                    linewidth=3,
                )
            )
            text = f"{self.model.config.id2label[cl.item()]}: {p[cl]:0.2f}"
            ax.text(
                xmin - 20,
                ymin - 50,
                text,
                fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.5),
            )

        plt.axis("off")
        plt.show()

    def detect_tables(self, input_img_path, resize_height=800, threshold_proba=0.7):
        start = time.perf_counter()
        # compo_json = json.load(open(compo_path, "r"))
        name = (
            input_img_path.split("/")[-1][:-4]
            if "/" in input_img_path
            else input_img_path.split("\\")[-1][:-4]
        )
        image, _, bboxes_scaled = self.detect(
            input_img_path, resize_height, threshold_proba
        )
        tables = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes_scaled.tolist()):
            table_id = i + 1
            table_height = ymax - ymin
            table_width = xmax - xmin
            table = {
                "id": table_id,
                "height": int(table_height),
                "width": int(table_width),
                "x1": int(xmin),
                "y1": int(ymin),
                "x2": int(xmax),
                "y2": int(ymax),
            }
            tables.append(table)
        ip_root = file.build_directory(pjoin(self.output_root, "tables"))
        # file.save_corners_json(pjoin(ip_root, name + ".json"), tables)
        with open(pjoin(ip_root, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"img_shape": [image.size[1], image.size[0], 3], "tables": tables},
                f,
                ensure_ascii=False,
                indent=4,
            )
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Table Detection Completed in %.3f s] Input: %s Output: %s"
                % (
                    time.perf_counter() - start,
                    input_img_path,
                    pjoin(ip_root, name + ".json"),
                )
            )
        return tables

    def extract_table_region(self, image, table, pad=20):
        x1, y1, x2, y2 = table["x1"], table["y1"], table["x2"], table["y2"]
        region = image.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))

        return region


# # Example usage:
# table_detector = TableDetector()
# image = Image.open("./Numbers-Table-Styles-Mac.jpg").convert("RGB")
# detected_tables = table_detector.detect_tables(image, threshold_proba=0.7)
# print(detected_tables)
