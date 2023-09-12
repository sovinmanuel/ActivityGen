from os.path import join as pjoin
import cv2
import os
from paddleocr import PaddleOCR
from activitygen.config.parameter_config import CONFIG_PARSER, load_config
import json

load_config("./configs/rico.cfg")
# load_config("./configs/uis_synthetic_log.cfg")
from activitygen.text_detector.detector import TextDetector
from activitygen.compo_detector.detector import CompoDetector
from activitygen.merge import merger

# CONFIG_DICT = dict(CONFIG_OBJ.items("MAIN"))


def resize_height_by_longest_edge(img_path, resize_length=1920):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def run_test():
    input_path_folder = "../../../_Datasets/RICO/combined/"
    output_root = "./rico_predictions"
    # High dpi (resize_length=2000)
    os.makedirs(output_root, exist_ok=True)
    data = json.load(open("./instances_test.json", "r"))

    input_imgs = [
        pjoin(input_path_folder, img["file_name"].split("/")[-1])
        for img in data["images"]
    ]
    input_imgs = sorted(
        input_imgs, key=lambda x: int(x.split("/")[-1][:-4])
    )  # sorted by index

    print("[Batch Init started]")

    textDetector = TextDetector(
        output_root,
        PaddleOCR(
            use_angle_cls=True,
            lang="en",
            # gpu=False,
            # det_db_thresh=0.3,
            # det_db_box_thresh=0.5,
            # det_db_unclip_ratio=2.0,
            # rec_char_type="en",
            # rec_image_shape="3, 32, 320",
            # rec_batch_num=30,
            # enable_mkldnn=True,
            # cls_thresh=0.9,
            # drop_score=0.5,
            show_log=False,
        ),
    )
    os.makedirs(pjoin(output_root, "merge"), exist_ok=True)
    compoDetector = CompoDetector(output_root)
    classifier = None
    # set the range of target inputs' indices
    start_index = 0  # 61728
    end_index = 100000
    for input_path_img in input_imgs:
        index = input_path_img.split("/")[-1][:-4]
        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break
        print(input_path_img)
        resized_height = resize_height_by_longest_edge(
            input_path_img, resize_length=800
        )
        textDetector.detect_text_paddle(input_path_img, show=False)
        img_name = input_path_img.split("/")[-1][:-4]
        # switch of the classification func
        compoDetector.detect_compos(
            input_path_img,
            resize_by_height=resized_height,
            show=False,
        )
        table_path = None

        template_matcher_path = None

        compo_path = pjoin(output_root, "compo", str(img_name) + ".json")
        ocr_path = pjoin(output_root, "ocr", str(img_name) + ".json")
        img_resize_shape, resized_image, data = merger.merge(
            input_path_img,
            compo_path,
            ocr_path,
            table_path,
            tm_path=template_matcher_path,
            merge_root=pjoin(output_root, "merge"),
            classifier=classifier,
            show=False,
        )


if __name__ == "__main__":
    # activitynames = run_batch_inference(
    #     "./../../../_Datasets/UIS Log Synthetic User Interface with Screenshots Log/Basic/Basic/scenario_0/Basic_10_Balanced/*.png",
    #     "User and Email Management App",
    #     "./data/output/uis_log_synthetic",
    # )
    # activitynames = run_batch_inference(
    #     "./data/inputs/mobile_input_test/*",
    #     "App",
    #     "./data/output/mobile_input_test",
    # )
    # activitynames = run_batch_inference(
    #     "./data/inputs/todo/*",
    #     "Todo App",
    #     "./data/output/todo",
    # )
    # activitynames = run_batch_inference(
    #     "./data/inputs/jira/*",
    #     "Jira",
    #     "./data/output/jira",
    # )
    run_test()
