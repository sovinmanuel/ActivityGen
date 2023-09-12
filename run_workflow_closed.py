import glob
import os
import time
from os.path import join as pjoin
from typing import List
import cv2
import numpy as np
from paddleocr import PaddleOCR
from activitygen.activity_name_generator.nlg_template_generator import (
    NLGTemplateActivityNameGenerator,
)
from activitygen.compo_classifier.resnet_classifier import ResNetClassifier
from activitygen.compo_classifier.sim_matcher import SimMatcher
from activitygen.compo_detector.tm_detector import TemplateMatcher
from activitygen.config.parameter_config import CONFIG_PARSER, load_config

# Load configuration
load_config("./configs/dropbox_workflow_closed.cfg")
# load_config("./configs/default.cfg")
from activitygen.text_detector.detector import TextDetector
from activitygen.compo_detector.detector import CompoDetector
from activitygen.compo_detector.table_detector import TableDetector
from activitygen.compo_classifier.classifier import ClassifierPipeline
from activitygen.merge import merger
from activitygen.activity_name_generator.generator import ActivityNameGenerator
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


# Function to resize image height based on the longest edge
def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


# Function for batch inference
def run_batch_inference(
    input_path_folder,
    appnames,
    output_root="data/output/mobile",
    sim_template_path="./templates/sc_templates",
):
    os.makedirs(output_root, exist_ok=True)

    print("[Batch Init started]")
    # Initialize components based on configuration settings
    if CONFIG_PARSER.getboolean("STEPS", "enable_ocr"):
        textDetector = TextDetector(
            output_root,
            PaddleOCR(
                use_angle_cls=False,
                lang="en",
                enable_mkldnn=True,
                show_log=False,
            ),
        )
        os.makedirs(pjoin(output_root, "ocr"), exist_ok=True)

    if CONFIG_PARSER.getboolean("STEPS", "enable_compo"):
        compoDetector = CompoDetector(output_root)

    template_matcher = None
    if CONFIG_PARSER.getboolean("STEPS", "enable_tm"):
        template_matcher = TemplateMatcher(output_root, "./templates/tm_templates/")

    clip_model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    resnet_classifier = ResNetClassifier("./models/classifier_model.pth")
    sim_matcher = None
    if (
        CONFIG_PARSER.getboolean("STEPS", "enable_merge")
        and CONFIG_PARSER.getboolean("STEPS", "enable_vism")
        and sim_template_path
    ):
        sim_matcher = SimMatcher(
            "./models/vism_model.pth",
            sim_template_path,
        )
    classifier = None
    if CONFIG_PARSER.getboolean("STEPS", "enable_classifier"):
        classifier = ClassifierPipeline(
            clip_model, clip_processor, resnet_classifier, sim_matcher
        )
    os.makedirs(pjoin(output_root, "merge"), exist_ok=True)

    # Initialize activity name generator based on configuration settings
    if CONFIG_PARSER.getboolean("STEPS", "enable_merge") and CONFIG_PARSER.getboolean(
        "STEPS", "enable_activity_name_generation"
    ):
        if CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "nlgtemplate":
            activityNameGen = NLGTemplateActivityNameGenerator()
        elif CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "extended":
            model_id = "nivos/flan-t5-base-activity-surrounding-summarize"
            lm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            lm_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            activityNameGen = ActivityNameGenerator(lm_model, lm_tokenizer)
        else:
            model_id = "nivos/pythia-410m-deduped-finetuned-final-activity-text-10epoch"
            lm_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            lm_model = AutoModelForCausalLM.from_pretrained(model_id)
            activityNameGen = ActivityNameGenerator(lm_model, lm_tokenizer)

    # Create directories for output paths
    activity_path_root = pjoin(output_root, "activity_name")
    os.makedirs(activity_path_root, exist_ok=True)
    activity_names_all = dict()

    import re

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        """
        return [atoi(c) for c in re.split(r"(\d+)", text)]

    # Process each input image
    input_images = glob.glob(input_path_folder)
    input_images.sort(key=natural_keys)
    for index, input_path_img in enumerate(input_images):
        start = time.perf_counter()
        print(f"[Generation started for: {input_path_img}]")
        appname = ""
        if isinstance(appnames, str):
            appname = appnames
        elif isinstance(appnames, list):
            appname = appnames[index]

        # Resize image
        resized_height = resize_height_by_longest_edge(
            input_path_img, resize_length=CONFIG_PARSER.getint("MAIN", "resize_length")
        )

        # Perform text detection using OCR
        if CONFIG_PARSER.getboolean("STEPS", "enable_ocr"):
            textDetector.detect_text_paddle(input_path_img, show=False)

        img_name = input_path_img.split("/")[-1][:-4]
        screenshotid = input_path_img.split("/")[-1]

        # Perform component detection
        if CONFIG_PARSER.getboolean("STEPS", "enable_compo"):
            compoDetector.detect_compos(
                input_path_img, resize_by_height=resized_height, show=False
            )

        table_path = None
        # Add table detection if needed
        # if table_detector and "excel" in appname.lower():
        #     table_detector.detect_tables(
        #         input_path_img, resize_height=resized_height, threshold_proba=0.7
        #     )
        #     table_path = pjoin(output_root, "tables", str(img_name) + ".json")

        template_matcher_path = None
        if template_matcher:
            template_matcher_path = pjoin(
                output_root, "template_matcher", str(img_name) + ".json"
            )
            template_matcher.detect_template_matches(
                input_path_img, template_matcher_path, resized_height=resized_height
            )

        activity_names = []

        compo_path = pjoin(output_root, "compo", str(img_name) + ".json")
        ocr_path = pjoin(output_root, "ocr", str(img_name) + ".json")

        # Merge components, OCR, and other information
        if (
            CONFIG_PARSER.getboolean("STEPS", "enable_ocr")
            and CONFIG_PARSER.getboolean("STEPS", "enable_compo")
            and CONFIG_PARSER.getboolean("STEPS", "enable_merge")
        ):
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

        # Generate activity names
        if CONFIG_PARSER.getboolean(
            "STEPS", "enable_merge"
        ) and CONFIG_PARSER.getboolean("STEPS", "enable_activity_name_generation"):
            activity_names = activityNameGen.generate_activity_names(
                appname,
                data,
                activity_path_root,
                img_name,
                resized_image,
                img_resize_shape,
            )
            activity_names_all[str(screenshotid)] = activity_names

        print("[Generation Completed in %.3f s]" % (time.perf_counter() - start))

    return activity_names_all


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
    activitynames = run_batch_inference(
        "./workflow/input/*",
        [],
        "./workflow/output/",
    )
    # activitynames = run_batch_inference(
    #     "./data/inputs/jira/*",
    #     "Jira",
    #     "./data/output/jira",
    # )
    # activitynames = run_batch_inference(
    #     "./data/inputs/synthetic_btn/*",
    #     "App",
    #     "./data/output/synthetic_btn",
    # )
