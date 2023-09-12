import cv2
import numpy as np
import time
from os.path import join as pjoin

from activitygen.config.parameter_config import CONFIG_PARSER
from . import file_utils as file
import json


class FeatureMatcher:
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

    def _homography_feature_matching(
        self, input_img_path, template_infos, default_threshold=0.6, resize_height=800
    ):
        org_img = cv2.imread(input_img_path)
        image = file.read_and_resize_image(input_img_path, resize_height)
        image_cv = file.convert_from_pil_to_cv2(image)
        imageGray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        results = []

        sift = cv2.SIFT_create()

        for template_info in template_infos:
            template_path = pjoin(
                self.template_dir_path, template_info["template_file_name"]
            )
            template = cv2.imread(template_path)
            org_img_height = org_img.shape[0]
            resize_ratio = resize_height / org_img_height
            template = cv2.resize(template, (0, 0), fx=resize_ratio, fy=resize_ratio)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_height, template_width, _ = template.shape

            keypoints_scene, descriptors_scene = sift.detectAndCompute(imageGray, None)
            keypoints_template, descriptors_template = sift.detectAndCompute(
                templateGray, None
            )

            flann = cv2.FlannBasedMatcher_create()
            matches = flann.knnMatch(descriptors_template, descriptors_scene, k=2)

            ratio_thresh = float(template_info["threshold"])
            if ratio_thresh < 0:
                ratio_thresh = default_threshold
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            if len(good_matches) < 4:
                continue
            obj = np.empty((len(good_matches), 2), dtype=np.float32)
            scene = np.empty((len(good_matches), 2), dtype=np.float32)
            for i, match in enumerate(good_matches):
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                obj[i, 0] = keypoints_template[query_idx].pt[0]
                obj[i, 1] = keypoints_template[query_idx].pt[1]
                scene[i, 0] = keypoints_scene[train_idx].pt[0]
                scene[i, 1] = keypoints_scene[train_idx].pt[1]

            H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)

            template_corners = np.float32(
                [
                    [0, 0],
                    [template_width, 0],
                    [template_width, template_height],
                    [0, template_height],
                ]
            ).reshape(-1, 1, 2)

            scene_corners = cv2.perspectiveTransform(template_corners, H)
            rect = cv2.boundingRect(scene_corners)
            x, y, w, h = rect

            result = {
                "template_name": template_info["template_name"],
                "negative_template": template_info["negative_template"],
                "activity_name": template_info["activity_name"],
                "bounding_box": (x, y, x + w, y + h),
            }
            results.append(result)

        return results, image

    def _bbox_feature_matching(
        self, input_img_path, template_infos, default_threshold=0.8, resize_height=800
    ):
        org_img = cv2.imread(input_img_path)
        image = file.read_and_resize_image(input_img_path, resize_height)
        image = file.convert_from_pil_to_cv2(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []

        sift = cv2.SIFT_create()

        for template_info in template_infos:
            template_path = pjoin(
                self.template_dir_path, template_info["template_file_name"]
            )
            template = cv2.imread(template_path)
            org_img_height = org_img.shape[0]
            resize_ratio = resize_height / org_img_height
            template = cv2.resize(template, (0, 0), fx=resize_ratio, fy=resize_ratio)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            keypoints_scene, descriptors_scene = sift.detectAndCompute(image, None)
            keypoints_template, descriptors_template = sift.detectAndCompute(
                template, None
            )

            flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
            matches = flann.knnMatch(descriptors_template, descriptors_scene, k=2)

            ratio_thresh = float(template_info["threshold"])
            if ratio_thresh < 0:
                ratio_thresh = default_threshold
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            scene_points = np.float32(
                [keypoints_scene[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            x, y, w, h = cv2.boundingRect(scene_points)

            result = {
                "template_name": template_info["template_name"],
                "negative_template": template_info["negative_template"],
                "activity_name": template_info["activity_name"],
                "bounding_box": (x, y, x + w, y + h),
            }
            results.append(result)

        return results, image

    def draw_matches(self, input_img_path, matches, resize_height=800):
        file.draw_matches(input_img_path, matches, resize_height=resize_height)

    def detect_feature_matches(self, input_img_path, resize_height=800, threshold=0.7):
        start = time.perf_counter()
        name = (
            input_img_path.split("/")[-1][:-4]
            if "/" in input_img_path
            else input_img_path.split("\\")[-1][:-4]
        )
        template_infos = self.read_json_lines_file(
            pjoin(self.template_dir_path, "metadata.jsonl")
        )
        results, image = self._bbox_feature_matching(
            input_img_path, template_infos, threshold, resize_height
        )
        # results, image = self._homography_feature_matching(
        #     input_img_path, template_infos, threshold, resize_height
        # )
        # results, image = self._basic_match_templates(
        #     input_img_path, threshold, resize_height
        # )
        # Todo: Built this into matching methods
        matches = []
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

        ip_root = file.build_directory(pjoin(self.output_root, "templates"))

        with open(pjoin(ip_root, name + "_fm.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"img_shape": file.get_image_shape(image), "matches": matches},
                f,
                ensure_ascii=False,
                indent=4,
            )
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Template Detection Completed in %.3f s] Input: %s Output: %s"
                % (
                    time.perf_counter() - start,
                    input_img_path,
                    pjoin(ip_root, name + "_fm.json"),
                )
            )

        return matches
