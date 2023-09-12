import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import glob
from os.path import join as pjoin
from typing import List
import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from activitygen.activity_name_generator.nlg_template_generator import (
    NLGTemplateActivityNameGenerator,
)
from activitygen.compo_classifier.resnet_classifier import ResNetClassifier
from activitygen.compo_classifier.sim_matcher import SimMatcher
from activitygen.compo_detector.tm_detector import TemplateMatcher
from activitygen.config.parameter_config import CONFIG_PARSER, load_config

load_config("./configs/default.cfg")
# load_config("./configs/uis_synthetic_log.cfg")
from activitygen.text_detector.detector import TextDetector
from activitygen.compo_detector.detector import CompoDetector
from activitygen.compo_detector.table_detector import TableDetector
from activitygen.compo_classifier.classifier import ClassifierPipeline
from activitygen.merge import merger
from activitygen.activity_name_generator.generator import (
    ActivityNameGenerator,
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


output_root = os.path.join("./api_data/output")
os.makedirs(output_root, exist_ok=True)
os.makedirs(os.path.join(output_root, "ocr"), exist_ok=True)
os.makedirs(os.path.join(output_root, "compo"), exist_ok=True)
os.makedirs(os.path.join(output_root, "merge"), exist_ok=True)
os.makedirs(os.path.join(output_root, "activity_name"), exist_ok=True)
os.makedirs(os.path.join("./api_data/input"), exist_ok=True)


textDetector = TextDetector(
    PaddleOCR(
        use_angle_cls=False,
        lang="en",
        enable_mkldnn=True,
        show_log=False,
    )
)

compoDetector = CompoDetector(output_root)
table_detector = None
table_detector = TableDetector(output_root)

# os.makedirs(tm_path_root, exist_ok=True)
template_matcher = None
template_matcher = TemplateMatcher(output_root, "./templates/tm_templates/")

clip_model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
resnet_classifier = ResNetClassifier("./models/classifier_model.pth")
sim_path = ("./templates/sc_templates",)
sim_matcher = SimMatcher(
    "./models/vism_model.pth",
    sim_path,
)
classifier = ClassifierPipeline(
    clip_model, clip_processor, resnet_classifier, sim_matcher
)
os.makedirs(pjoin(output_root, "merge"), exist_ok=True)

model_id = "JohnPedda/pythia-410m-deduped-finetuned-final-activity-text-5epoch"
lm_tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    # padding_side="left",
)
lm_model = AutoModelForCausalLM.from_pretrained(model_id)

# model_id = "JohnPedda/flan-t5-base-activity-surrounding-summarize"
# lm_tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     # padding_side="left",
# )
# lm_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

activityNameGen = ActivityNameGenerator(lm_model, lm_tokenizer)

# activityNameGen = NLGTemplateActivityNameGenerator()


@app.post("/predict")
async def predict_activity_names(file: UploadFile = File(...), appname: str = ""):
    start = time.perf_counter()
    # Save the uploaded file to disk
    input_path_img = os.path.join("./api_data/input", file.filename)
    with open(input_path_img, "wb") as f:
        f.write(await file.read())

    print("[Inference started]")

    activity_names_all = []

    appname = ""

    resized_height = resize_height_by_longest_edge(input_path_img, resize_length=1920)
    textDetector.detect_text_paddle(input_path_img, output_root, show=False)
    img_name = input_path_img.split("/")[-1][:-4]
    # switch of the classification func
    compoDetector.detect_compos(
        input_path_img,
        resize_by_height=resized_height,
        show=False,
    )
    table_path = None
    if table_detector and "excel" in appname.lower():
        table_detector.detect_tables(
            input_path_img, resize_height=resized_height, threshold_proba=0.7
        )
        table_path = pjoin(output_root, "tables", str(img_name) + ".json")

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
    activity_path_root = pjoin(output_root, "activity_name")
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
    activity_names = activityNameGen.generate_activity_names(
        appname, data, activity_path_root, img_name, resized_image, img_resize_shape
    )
    activity_names_all.append(activity_names)
    print("[Inference Completed in %.3f s]" % (time.perf_counter() - start))
    return {"activity_names": activity_names_all}


# @app.post("/templates/add")
# async def add_templates(directory: str):
#     files = os.listdir(directory)
#     for file_name in files:
#         if file_name.endswith(".png") or file_name.endswith(".jpg"):
#             image_path = os.path.join(directory, file_name)
#             matching_method = "default"  # Change as per your requirement
#             template_db.add_template(file_name, image_path, matching_method)
#             # Optionally, save the image to the file system
#             ImageUtils.save_image(image_path, await File(file=image_path).read())
#     return {"message": "Templates added successfully."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
