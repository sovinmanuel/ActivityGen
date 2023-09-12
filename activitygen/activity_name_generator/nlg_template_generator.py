import re
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cv2
from os.path import join as pjoin
import os
import openai

from activitygen.activity_name_generator.activity_db import ActivityNameDatabaseManager
from activitygen.activity_name_generator.similarity_db import CategoryCollection
from activitygen.config.parameter_config import CONFIG_PARSER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NLGTemplateActivityNameGenerator:
    def __init__(self):
        # Todo: DB wieder adden
        # self.db = DatabaseManager()
        self.sim_db = CategoryCollection()

    def initialize(self):
        self.model = self.model.to(device)

    def generate_activity_names(
        self, app_name, components, ag_path, name, resized_image, img_resize_shape
    ):
        start = time.perf_counter()
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print("[Activity Name Generation Started]")

        resized_image_copy = resized_image.copy()
        all_activity_names_full = []
        all_activity_names = []
        new_prompts_info = []
        new_bboxes = []
        existing_prompts_info = []
        existing_bboxes = []
        existing_activity_names = []

        to_generate = self._get_entries(app_name, components)

        for entry in to_generate:
            existing_activity_name = entry["existing_activity_name"]
            prompt_info = entry["prompt_info"]
            # print("_________")
            # print("existing_activity_name")
            # print(prompt_info)
            if existing_activity_name:
                # print(existing_activity_name)
                existing_prompts_info.append(prompt_info)
                existing_bboxes.append(entry["bbox"])
                existing_activity_names.append(existing_activity_name)
                continue

            new_prompts_info.append(prompt_info)
            new_bboxes.append(entry["bbox"])

        if new_prompts_info:
            generated_text = self._infer_text(new_prompts_info)
            entries = zip(new_bboxes, generated_text, new_prompts_info)
            self._process_new_entries(
                entries, resized_image_copy, all_activity_names_full, all_activity_names
            )
        elif CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print("No new activities")

        existing_entries = zip(
            existing_bboxes, existing_activity_names, existing_prompts_info
        )
        self._process_existing_entries(
            existing_entries,
            resized_image_copy,
            all_activity_names_full,
            all_activity_names,
        )

        cv2.imwrite(pjoin(ag_path, str(name) + ".jpg"), resized_image_copy)
        cv2.imwrite(pjoin(ag_path, str(name) + "_resized.jpg"), resized_image)
        full_json = {
            "img_resize_shape": img_resize_shape,
            "activity_names": all_activity_names_full,
        }
        self._save_elements(
            pjoin(ag_path, str(name) + "_full.json"),
            full_json,
        )
        self._save_elements(
            pjoin(ag_path, str(name) + ".json"),
            {
                "img_resize_shape": img_resize_shape,
                "activity_names": all_activity_names,
            },
        )
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Activity Name Generation Completed in %.3f s]"
                % (time.perf_counter() - start)
            )
        return full_json

    def _infer_text(self, prompt_infos):
        generated_texts = []
        for prompt_info in prompt_infos:
            compo_class = prompt_info["Component Class"]
            generated_text = ""
            if compo_class == "Button":
                generated_text = self.generate_btn_activity(prompt_info)
            elif "Checkbox" in compo_class:
                generated_text = self.generate_checkbox_activity(prompt_info)
                generated_text = self.generate_checkbox_activity(prompt_info)
            elif "EditText" == compo_class:
                generated_text = self.generate_edittext_activity(prompt_info)
            generated_texts.append(generated_text)

        return generated_texts

    def generate_btn_activity(self, prompt_info):
        inner_text = prompt_info["Inner Text"]
        return f"Click on the Button '{inner_text}'"

    def generate_edittext_activity(self, prompt_info):
        inner_text = prompt_info["Inner Text"]
        if inner_text:
            return f"Edit input '{inner_text}'"
        else:
            return f"Edit input"

    def generate_checkbox_activity(self, prompt_info):
        prefix = "Check"
        if prompt_info["checkbox_state"] == "checked":
            prefix = "Uncheck"
        inner_text = prompt_info["Surrounding Text"]
        return f"{prefix}"
        # return f"{prefix} '{inner_text}'"

    def _process_new_entries(
        self, entries, resized_image, all_activity_names_full, all_activity_names
    ):
        for entry in entries:
            # activity_name = self._determine_match(entry[1])
            activity_name = entry[1]
            if activity_name != "":
                bbox = entry[0]
                prompt = entry[2]
                all_activity_names_full.append(
                    {
                        "activity_name": activity_name,
                        "bbox": bbox,
                        "prompt": prompt,
                        "existing": "false",
                    }
                )
                all_activity_names.append(activity_name)
                self.visualize_element(resized_image, bbox, activity_name)
                # self.db.insert_data(
                #     prompt["Context (App Name)"],
                #     prompt["Context (App Category)"],
                #     prompt["Component Class"],
                #     prompt["Inner Text"],
                #     prompt["Surrounding Text"],
                #     activity_name,
                # )

    def _process_existing_entries(
        self,
        existing_entries,
        resized_image,
        all_activity_names_full,
        all_activity_names,
    ):
        for existing_entry in existing_entries:
            activity_name = existing_entry[1]
            bbox = existing_entry[0]
            prompt = existing_entry[2]
            all_activity_names_full.append(
                {
                    "activity_name": activity_name,
                    "bbox": bbox,
                    "prompt": prompt,
                    "existing": "true",
                }
            )
            all_activity_names.append(activity_name)
            self.visualize_element(resized_image, bbox, activity_name)

    def is_long_text(self, text):
        words = text.split()
        return len(words) > 2

    def _has_ancestor_block_or_compo(self, compo_id, compos):
        def has_ancestor(compo_id):
            if "parent" in compos[compo_id]:
                parent_id = compos[compo_id]["parent"]
                parent_class = compos[parent_id]["class"]
                if parent_class == "Block" or parent_class == "Compo":
                    return parent_id
                else:
                    return has_ancestor(parent_id)
            else:
                return None

        return has_ancestor(compo_id)

    def get_all_text_from_id(self, compo_id, compos):
        text_content = ""
        TABU_LIST = ["Button", "EditText"]
        current_compo = compos[compo_id]
        if "children" in current_compo:
            for child_compo_id in current_compo["children"]:
                child_compo = compos[child_compo_id]
                child_compo_parent = None
                if "parent" in child_compo:
                    child_compo_parent = compos[child_compo["parent"]]
                if (
                    child_compo_parent
                    and child_compo_parent["class"] not in TABU_LIST
                    and child_compo["class"] == "Text"
                    and "text_content" in child_compo
                ):
                    text_content += child_compo["text_content"] + "; "
                else:
                    child_text = self.get_all_text_from_id(child_compo_id, compos)
                    if child_text:
                        text_content += child_text

        return text_content.strip()

    def _get_entries(self, app_name, context):
        entries = []
        compos = context["compos"]
        raw_texts = app_name + "; " + context["raw_texts"]

        query_result = self.sim_db.query_categories(raw_texts)
        category = str(query_result["metadatas"][0][0]["category"])  # type:ignore

        if app_name.lower() == "app":
            app_name = category + " App"

        # compos = self.filter_out_irrelevant_compo(compos)
        for compo in compos:
            TABU_LIST = ["Image", "Block", "Compo"]
            comp_class = str(compo["class"])
            if comp_class.lower() in [x.lower() for x in TABU_LIST]:
                continue
            if (
                comp_class == "Text"
                and compo["text_content"]
                and (self.is_long_text(compo["text_content"]) or "parent" in compo)
            ):
                continue

            # if (
            #     (comp_class == "Text" or comp_class == "Image" or "Icon" in comp_class)
            #     and "parent" in compo
            #     and (
            #         compos[compo["parent"]]["class"] == "Button"
            #         or compos[compo["parent"]]["class"] == "EditText"
            #     )
            # ):
            #     continue

            # if (comp_class == "Block" or comp_class == "Compo") and "children" in compo:
            #     if len(compo["children"]) > 2:
            #         continue

            # Check Text
            if comp_class == "Text" and compo["text_content"] and not "parent" in compo:
                inner_text = str(compo["text_content"])
            else:
                inner_text = ""
                if "children" in compo:
                    for child_id in compo["children"]:
                        child = compos[child_id]
                        if child["class"] == "Text":
                            inner_text += " " + str(child["text_content"])

            # Check surrounding text
            surrounding_text = ""
            block_parent_id = self._has_ancestor_block_or_compo(compo["id"], compos)
            if block_parent_id:
                surrounding_text = self.get_all_text_from_id(block_parent_id, compos)

            # Check Icons
            if "Icon" in comp_class:
                inner_text = comp_class.split("-")[-1]
                comp_class = "Icon"

            if "checkbox_state" in compo:
                prompt_info = {
                    "Context (App Name)": app_name,
                    "Context (App Category)": category,
                    "Component Class": comp_class,
                    "Inner Text": inner_text,
                    "Surrounding Text": surrounding_text,
                    "checkbox_state": compo["checkbox_state"],
                }
            else:
                prompt_info = {
                    "Context (App Name)": app_name,
                    "Context (App Category)": category,
                    "Component Class": comp_class,
                    "Inner Text": inner_text,
                    "Surrounding Text": surrounding_text,
                }

            existing_activity_name = None
            # existing_activity_name = self.db.check_entry_exists(
            #     app_name, category, comp_class, inner_text, surrounding_text
            # )
            entries.append(
                {
                    "bbox": {
                        "x1": compo["position"]["x1"],
                        "y1": compo["position"]["y1"],
                        "x2": compo["position"]["x2"],
                        "y2": compo["position"]["y2"],
                    },
                    "prompt_info": prompt_info,
                    "existing_activity_name": existing_activity_name,
                }
            )

        return entries

    def _save_elements(self, output_file, activity_names):
        json.dump(activity_names, open(output_file, "w"), indent=4)

    def visualize_element(self, img, box, text, color=(0, 0, 255), line=1, show=False):
        cv2.rectangle(img, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, line)  # type: ignore
        cv2.putText(
            img,
            text,
            (box["x1"], box["y1"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            255,
            thickness=3,
        )
