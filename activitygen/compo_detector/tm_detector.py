import cv2
import numpy as np
import time
from os.path import join as pjoin

from activitygen.config.parameter_config import CONFIG_PARSER

from . import file_utils as file
import json
from imutils.object_detection import non_max_suppression


class TemplateMatcher:
    def __init__(self, output_root, template_dir_path):
        self.output_root = output_root
        self.template_dir_path = template_dir_path

    def read_json_lines_file(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    # print(f"Skipping line: {line.strip()}. Error: {str(e)}")
                    pass
        return data

    def _basic_match_templates(
        self, input_img_path, template_infos, default_threshold=0.8, resize_height=800
    ):
        org_img = cv2.imread(input_img_path)
        image = file.read_and_resize_image(input_img_path, resize_height)
        image_cv = file.convert_from_pil_to_cv2(image)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        results = []
        for template_info in template_infos:
            template_path = pjoin(
                self.template_dir_path, template_info["template_file_name"]
            )
            template = cv2.imread(template_path)
            org_img_height = org_img.shape[0]
            resize_ratio = resize_height / org_img_height
            template = cv2.resize(template, (0, 0), fx=resize_ratio, fy=resize_ratio)
            # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            height, width, _ = template.shape
            # Todo: Check again
            result = cv2.matchTemplate(image_cv, template, cv2.TM_CCOEFF_NORMED)
            threshold = float(template_info["threshold"])
            if threshold < 0:
                threshold = default_threshold
            (yCoords, xCoords) = np.where(result >= threshold)
            rects = []
            for x, y in zip(xCoords, yCoords):
                rects.append((x, y, x + width, y + height))
            pick = non_max_suppression(np.array(rects))
            for x1, y1, x2, y2 in pick:
                result = {
                    "template_name": template_info["template_name"],
                    "negative_template": template_info["negative_template"],
                    "activity_name": template_info["activity_name"],
                    "bounding_box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
                results.append(result)
        return results, image

    def draw_matches(self, input_img_path, matches, resize_height=800):
        file.draw_matches(input_img_path, matches, resize_height=resize_height)

    def detect_template_matches(
        self, input_img_path, template_matcher_path, resized_height=800, threshold=0.7
    ):
        start = time.perf_counter()
        name = (
            input_img_path.split("/")[-1][:-4]
            if "/" in input_img_path
            else input_img_path.split("\\")[-1][:-4]
        )
        template_infos = self.read_json_lines_file(
            pjoin(self.template_dir_path, "metadata.jsonl")
        )
        results, image = self._basic_match_templates(
            input_img_path, template_infos, threshold, resized_height
        )
        matches = []
        # Todo: Built this into matching methods
        for i, result in enumerate(results):
            match_id = i + 1
            bbox = result["bounding_box"]
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            height = ymax - ymin
            width = xmax - xmin
            match = {
                "id": match_id,
                "template_name": result["template_name"],
                "negative_template": result["negative_template"],
                "activity_name": result["activity_name"],
                "height": int(height),
                "width": int(width),
                "x1": int(xmin),
                "y1": int(ymin),
                "x2": int(xmax),
                "y2": int(ymax),
            }
            matches.append(match)

        tm_root = file.build_directory(pjoin(self.output_root, "template_matcher"))

        with open(pjoin(tm_root, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"img_shape": file.get_image_shape(image), "matches": matches},
                f,
                ensure_ascii=False,
                indent=4,
            )
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Template Detection Completed in %.3f s] Input: %s Output: %s"
                % (time.perf_counter() - start, input_img_path, template_matcher_path)
            )

        return matches
