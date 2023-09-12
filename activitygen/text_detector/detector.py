import cv2
import json
import numpy as np
import time
from os.path import join as pjoin
from paddleocr import PaddleOCR

from activitygen.config.parameter_config import CONFIG_PARSER
from .text import Text

import activitygen.compo_detector.file_utils as file


class TextDetector:
    def __init__(self, output_root, model=None):
        self.paddle_model = model
        self.output_root = output_root

    def save_detection_json(self, file_path, texts, img_shape):
        with open(file_path, "w") as f_out:
            output = {"img_shape": img_shape, "texts": []}
            for text in texts:
                c = {
                    "id": text.id,
                    "content": text.content,
                    "x1": text.location["x1"],
                    "y1": text.location["y1"],
                    "x2": text.location["x2"],
                    "y2": text.location["y2"],
                    "width": text.width,
                    "height": text.height,
                }
                output["texts"].append(c)
            json.dump(output, f_out, indent=4)

    def visualize_texts(
        self, org_img, texts, shown_resize_height=None, show=False, write_path=None
    ):
        img = org_img.copy()
        for text in texts:
            text.visualize_element(img, line=2)

        img_resize = img
        if shown_resize_height is not None:
            img_resize = cv2.resize(
                img,
                (
                    int(shown_resize_height * (img.shape[1] / img.shape[0])),
                    shown_resize_height,
                ),
            )

        if show:
            cv2.imshow("texts", img_resize)
            cv2.waitKey(0)
            cv2.destroyWindow("texts")
        if write_path is not None:
            cv2.imwrite(write_path, img)

    def text_cvt_orc_format_paddle(self, paddle_result):
        texts = []
        for idx in range(len(paddle_result)):
            res = paddle_result[idx]
            for line in res:
                points = np.array(line[0])
                location = {
                    "x1": int(min(points[:, 0])),
                    "y1": int(min(points[:, 1])),
                    "x2": int(max(points[:, 0])),
                    "y2": int(max(points[:, 1])),
                }
                content = line[1][0]
                texts.append(Text(idx, content, location))
        return texts

    def detect_text(
        self,
        input_file="../data/input/30800.jpg",
        show=False,
    ):
        import pytesseract

        start = time.perf_counter()
        name = input_file.split("/")[-1][:-4]
        ocr_root = file.build_directory(pjoin(self.output_root, "ocr"))
        img = cv2.imread(input_file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        texts_raw = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = img[y : y + h, x : x + w]

            # Apply OCR on the cropped image
            text = (
                pytesseract.image_to_string(cropped)
                .replace("\x0c", "")
                .replace("\n", "")
            )
            location = {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
            }
            if text:
                texts_raw.append({"text": text, "location": location})

        texts = []
        for id, text_raw in enumerate(texts_raw):
            texts.append(Text(id, text_raw["text"], text_raw["location"]))

        self.visualize_texts(
            img,
            texts,
            shown_resize_height=800,
            show=show,
            write_path=pjoin(ocr_root, name + ".png"),
        )
        self.save_detection_json(pjoin(ocr_root, name + ".json"), texts, img.shape)
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Text Detection Completed in %.3f s] Input: %s Output: %s"
                % (
                    time.perf_counter() - start,
                    input_file,
                    pjoin(ocr_root, name + ".json"),
                )
            )

    def detect_text_paddle(
        self,
        input_file="../data/input/30800.jpg",
        show=False,
    ):
        start = time.perf_counter()
        ocr_root = file.build_directory(pjoin(self.output_root, "ocr"))
        name = input_file.split("/")[-1][:-4]
        img = cv2.imread(input_file)

        if self.paddle_model is None:
            self.paddle_model = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                gpu=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=2.0,
                rec_char_type="en",
                rec_image_shape="3, 32, 320",
                rec_batch_num=30,
                enable_mkldnn=True,
                cls_thresh=0.9,
                drop_score=0.5,
                show_log=False,
            )
        result = self.paddle_model.ocr(input_file, cls=False)
        texts = self.text_cvt_orc_format_paddle(result)

        self.visualize_texts(
            img,
            texts,
            shown_resize_height=800,
            show=show,
            write_path=pjoin(ocr_root, name + ".png"),
        )
        self.save_detection_json(pjoin(ocr_root, name + ".json"), texts, img.shape)
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Text Detection Completed in %.3f s] Input: %s Output: %s"
                % (
                    time.perf_counter() - start,
                    input_file,
                    pjoin(ocr_root, name + ".json"),
                )
            )
