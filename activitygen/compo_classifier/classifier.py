import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import torch

import cv2
from activitygen.compo_classifier.resnet_classifier import ResNetClassifier
from activitygen.compo_classifier.sim_matcher import SimMatcher
from activitygen.merge.element import Element
from typing import List
from activitygen.config.parameter_config import CONFIG_PARSER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ClassifierPipeline:
    def __init__(
        self,
        clip_model,
        clip_processor,
        resnet_classifier: ResNetClassifier,
        sim_matcher: SimMatcher | None = None,
    ):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.resnet_classifier = resnet_classifier
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.clip_model.to(device)
        self.sim_matcher = sim_matcher

    def check_is_link(self, element):
        return True

    def return_checkbox_state(self, checkbox_element_image):
        state = "checked"
        if self.sim_matcher:
            image_paths, _ = self.sim_matcher.return_checkbox_state(
                checkbox_element_image
            )
            if image_paths[0] == 0:
                state = None
            elif "unchecked" in image_paths[0]:
                state = "unchecked"
            else:
                state = "checked"

        return state

    def predict(self, imgs, elements: List[Element]):
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print("[Classification Started]")

        icons_names = [
            "Search",
            "Menu",
            "Options",
            "Home",
            "Heart",
            "Shopping Bag",
            "Profile",
            "Email",
            "Phone",
            "Settings",
            "Lock",
            "Chat",
            "Calendar",
            "Download",
            "Upload",
            "Checkmark",
            "Close",
            "Cancel",
            "Refresh",
            "Play",
            "Pause",
            "Stop",
            "Next",
            "Previous",
            "Add",
            "Minus",
            "Edit",
            "Trash",
            "Notification",
            "Warning",
            "Info",
            "Share",
            "Globe",
            "Bookmark",
            "Compass",
            "Eye",
            "Location",
            "Clock",
            "File",
            "Music",
            "Camera",
            "Shield",
            "Puzzle Piece",
            "Smile",
            "Sad",
            "Weather",
            "Flag",
            "Health",
            "Wifi",
            "Printer",
            "Wrench",
            "Clipboard",
            "Credit Card",
            "Infinity",
            "Magnifying Glass",
            "Paintbrush",
            "Calendar",
            "Cogwheel",
            "Barcode",
            "Battery",
            "Link",
            "Pointer",
            "Book",
            "Microphone",
            "Speaker",
            "Louder",
            "Key",
            "Time",
            "Wallet",
            "Star",
            "Like",
        ]

        for i, image in enumerate(imgs):
            if elements[i].is_custom:
                continue
            has_non_text_children = False
            for child in elements[i].children:
                if child.category != "Text":
                    has_non_text_children = True
                    break

            has_text_children = False
            for child in elements[i].children:
                if child.category == "Text":
                    has_text_children = True
                    break

            # print("[ResNet started]")
            if elements[i].category != "Text" and elements[i].category != "Block":
                new_class, probability = self.resnet_classifier.classify_image(image)
                if probability >= CONFIG_PARSER.getfloat(
                    "MAIN", "min_resnet_compo_classifier_proba"
                ):
                    elements[i].category = new_class
                # cv2.imshow(elements[i].category, image)
                # cv2.waitKey()

            # Classify Icons
            if elements[i].category == "Icon":
                # if elements[i].category == "Icon" and not has_text_children:
                # print(elements[i].id)
                text_descriptions = [
                    f"This is an icon for {icons_name}" for icons_name in icons_names
                ]
                # text_descriptions.append("This is a non-graphic image")
                inputs = self.clip_processor(
                    text=text_descriptions,
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                results = self.clip_model(**inputs)
                logits_per_image = results[
                    "logits_per_image"
                ]  # this is the image-text similarity score
                probs = (
                    logits_per_image.softmax(dim=1).detach().cpu().numpy()
                )  # we can take the softmax to get the label probabilities
                predicted_icon = icons_names[np.argmax(probs)]
                # print(predicted_icon)
                elements[i].category = "Icon-" + predicted_icon
                del inputs
                del results
                del logits_per_image
                del probs
                torch.cuda.empty_cache()

            # print("[SimMatcher started]")
            # (
            #     image_paths,
            #     cosine_similarities,
            # ) = self.sim_matcher.find_most_similar_templates(image)
            # cosine_similarity = cosine_similarities[0]
            # if cosine_similarity > 0.95:
            #     print(cosine_similarities)
            #     # cv2.imshow(str(image_paths[0]), image)
            #     # cv2.waitKey()

            if self.sim_matcher:
                (
                    most_similar_template_name,
                    activity_name,
                ) = self.sim_matcher.detect_template_matches(image, elements[i])
                if most_similar_template_name:
                    elements[i].category = most_similar_template_name
                    elements[i].activity_name = activity_name
                    elements[i].is_custom = True
                    elements[i].is_vism = True

            if elements[i].category == "Checkbox":
                state = self.return_checkbox_state(image)
                if state:
                    # elements[i].category = "Checkbox-" + state
                    elements[i].checkbox_state = state

            # if elements[i].category == "Text":
