import cv2
import numpy as np
import time
from os.path import join as pjoin
from activitygen.config.parameter_config import CONFIG_PARSER

import file_utils as file
import json
from imutils import contours
from paddleocr import PaddleOCR


class CellDetector:
    def __init__(
        self,
        output_root,
        paddle_model: PaddleOCR,
        crop_top_size=212,
        crop_bottom_ratio=0.1,
        crop_left_size=5,
    ):
        self.output_root = output_root
        self.paddle_model = paddle_model
        self.crop_top_size = crop_top_size
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_left_size = crop_left_size

    def crop_row(self, image, row_y1, row_y2):
        if image is None:
            print("Error loading image")
            return None

        image_height = image.shape[0]

        if row_y1 < 0 or row_y1 >= image_height or row_y2 < 0 or row_y2 >= image_height:
            print("Invalid row coordinates")
            return None

        cropped_row = image[row_y1 : row_y2 + 1, :]

        cv2.imshow("Cropped Row", cropped_row)
        cv2.waitKey(0)

        return cropped_row

    def crop_column(self, image, col_x1, col_x2):
        if image is None:
            print("Error loading image")
            return None

        image_width = image.shape[1]

        if col_x1 < 0 or col_x1 >= image_width or col_x2 < 0 or col_x2 >= image_width:
            print("Invalid column coordinates")
            return None

        cropped_column = image[:, col_x1 : col_x2 + 1]

        cv2.imshow("Cropped Column", cropped_column)
        cv2.waitKey(0)

        return cropped_column

    def crop_cell(self, image, row_y1, row_y2, col_x1, col_x2, show=False):
        if image is None:
            print("Error loading image")
            return None

        image_height, image_width, _ = image.shape

        if row_y1 < 0 or row_y1 >= image_height or row_y2 < 0 or row_y2 >= image_height:
            print("Invalid row coordinates")
            return None

        if col_x1 < 0 or col_x1 >= image_width or col_x2 < 0 or col_x2 >= image_width:
            print("Invalid column coordinates")
            return None

        cropped_cell = image[row_y1 : row_y2 + 1, col_x1 : col_x2 + 1, :]

        if show:
            cv2.imshow("Cropped Cell", cropped_cell)
            cv2.waitKey(0)
        return cropped_cell

    def get_contours(self, input_img_path):
        image = cv2.imread(input_img_path)

        height = image.shape[0]
        crop_bottom_size = int(self.crop_bottom_ratio * height)
        image = image[
            self.crop_top_size : (height - crop_bottom_size), self.crop_left_size :
        ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 20)
        dilated_edges = cv2.dilate(edges, None, iterations=1)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        detect_horizontal = cv2.morphologyEx(
            dilated_edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=5
        )

        horizontal_cnts = cv2.findContours(
            detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        horizontal_cnts = (
            horizontal_cnts[0] if len(horizontal_cnts) == 2 else horizontal_cnts[1]
        )
        (horizontal_cnts, _) = contours.sort_contours(
            horizontal_cnts, method="top-to-bottom"
        )

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        detect_vertical = cv2.morphologyEx(
            dilated_edges, cv2.MORPH_OPEN, vertical_kernel, iterations=5
        )
        vertical_cnts = cv2.findContours(
            detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        vertical_cnts = (
            vertical_cnts[0] if len(vertical_cnts) == 2 else vertical_cnts[1]
        )
        (vertical_cnts, _) = contours.sort_contours(
            vertical_cnts, method="left-to-right"
        )

        return horizontal_cnts, vertical_cnts, image

    def get_cell_pos(self, mouse_position, horizontal_cnts, vertical_cnts):
        cell_x1 = -1
        cell_y1 = -1
        cell_x2 = -1
        cell_y2 = -1

        for i, _ in enumerate(horizontal_cnts[:-1]):
            x1_1, y1_1, w_1, h_1 = cv2.boundingRect(horizontal_cnts[i])
            x2_1 = x1_1 + w_1
            y2_1 = y1_1 + h_1

            x1_2, y1_2, w_2, h_2 = cv2.boundingRect(horizontal_cnts[i + 1])
            x2_2 = x1_2 + w_2
            y2_2 = y1_2 + h_2

            if y1_1 <= mouse_position[1] and mouse_position[1] < y2_2:
                cell_y1 = y2_1
                cell_y2 = y1_2

        for i, _ in enumerate(vertical_cnts[:-1]):
            x1_1, y1_1, w_1, h_1 = cv2.boundingRect(vertical_cnts[i])
            x2_1 = x1_1 + w_1
            y2_1 = y1_1 + h_1

            x1_2, y1_2, w_2, h_2 = cv2.boundingRect(vertical_cnts[i + 1])
            x2_2 = x1_2 + w_2
            y2_2 = y1_2 + h_2

            if x1_1 <= mouse_position[0] and mouse_position[0] < x2_2:
                cell_x1 = x2_1
                cell_x2 = x1_2

        return cell_x1, cell_y1, cell_x2, cell_y2

    def detect_cell(self, input_img_path, mouse_position):
        start = time.perf_counter()
        name = (
            input_img_path.split("/")[-1][:-4]
            if "/" in input_img_path
            else input_img_path.split("\\")[-1][:-4]
        )
        horizontal_cnts, vertical_cnts, image = self.get_contours(input_img_path)

        cell_x1, cell_y1, cell_x2, cell_y2 = self.get_cell_pos(
            mouse_position, horizontal_cnts, vertical_cnts
        )

        first_x1, _, _, _ = cv2.boundingRect(vertical_cnts[0])
        row_index_cell = self.crop_cell(image, cell_y1, cell_y2, 0, first_x1, False)

        _, first_y1, _, _ = cv2.boundingRect(horizontal_cnts[0])
        col_index_cell = self.crop_cell(image, 0, first_y1, cell_x1, cell_x2, False)
        row_index_name = self.paddle_model.ocr(row_index_cell)[0][0][1][0]
        col_index_name = self.paddle_model.ocr(col_index_cell)[0][0][1][0]

        cell_cropped_image = self.crop_cell(
            image, cell_y1, cell_y2, cell_x1, cell_x2, False
        )
        cell_data = self.paddle_model.ocr(cell_cropped_image)[0][0][1][0]

        cells = []
        cell = {
            # "relative_row_index": ,
            "row_index_name": row_index_name,
            "col_index_name": col_index_name,
            "cell_data": cell_data,
            "x1": int(cell_x1),
            "y1": int(cell_y1),
            "x2": int(cell_x2),
            "y2": int(cell_y2),
        }
        cells.append(cell)

        cd_root = file.build_directory(pjoin(self.output_root, "cell_detection"))

        with open(pjoin(cd_root, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"img_shape": file.get_image_shape(image), "cells": cells},
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
                    pjoin(cd_root, name + ".json"),
                )
            )

        return cells
