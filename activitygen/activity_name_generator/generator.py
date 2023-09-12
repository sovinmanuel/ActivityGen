import re
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cv2
from os.path import join as pjoin
import os
import openai
from activitygen.activity_name_generator.extended_activity_db import (
    ActivityNameDatabaseManagerExtended,
)
from activitygen.config.parameter_config import CONFIG_PARSER
from activitygen.activity_name_generator.activity_db import ActivityNameDatabaseManager
from activitygen.activity_name_generator.similarity_db import CategoryCollection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActivityNameGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.model.to(device)
        self.tokenizer = tokenizer
        self.model.eval()
        # Todo: DB wieder adden
        if CONFIG_PARSER.get(
            "MAIN", "activity_generation_mode"
        ) == "basic" and CONFIG_PARSER.getboolean("MAIN", "save_activity_names"):
            self.db = ActivityNameDatabaseManager()
        elif CONFIG_PARSER.get(
            "MAIN", "activity_generation_mode"
        ) == "extended" and CONFIG_PARSER.getboolean("MAIN", "save_activity_names"):
            self.db = ActivityNameDatabaseManagerExtended()
        self.sim_db = CategoryCollection()

        # self.api_key = os.environ.get("OPENAI_API_KEY")
        # openai.api_key = self.api_key
        # self.openai_model_name = "gpt-3.5-turbo"

    def generate_activity_names(
        self, app_name, components, ag_path, name, resized_image, img_resize_shape
    ):
        start = time.perf_counter()
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print("[Activity Name Generation Started]")

        resized_image_copy = resized_image.copy()
        all_activity_names = []
        existing_entries = []
        new_entries = []
        new_prompts = []

        to_generate = self._get_entries(app_name, components)

        for entry in to_generate:
            if entry["existing_activity_name"]:
                existing_entries.append(entry)
                continue
            prompt_info = entry["prompt_info"]
            prompt = "Continue writing the following text.\n\n"
            for key, value in prompt_info.items():
                prompt += f"{key}: {value}, "
            prompt += f"Activity Name:"
            new_prompts.append(prompt)
            new_entries.append(entry)
        if new_prompts:
            generated_text = self._infer_text(new_prompts)
            all_activity_names.extend(
                self._process_new_entries(
                    new_entries, generated_text, resized_image_copy
                )
            )
        elif CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print("No new activities")

        if existing_entries:
            all_activity_names.extend(
                self._process_existing_entries(
                    existing_entries,
                    resized_image_copy,
                )
            )

        cv2.imwrite(pjoin(ag_path, str(name) + ".jpg"), resized_image_copy)
        # cv2.imwrite(pjoin(ag_path, str(name) + "_resized.jpg"), resized_image)
        full_json = {
            "img_resize_shape": img_resize_shape,
            "activity_names": all_activity_names,
        }
        self._save_elements(
            pjoin(ag_path, str(name) + ".json"),
            full_json,
        )
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Activity Name Generation Completed in %.3f s]"
                % (time.perf_counter() - start)
            )
        return full_json

    def _infer_text(self, prompts):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        generation_params = {
            "max_length": 70,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
            "top_k": 100,
            "top_p": 1,
            "temperature": 0.2,
            "num_return_sequences": 1,
            "repetition_penalty": 1.2,
        }
        encodings = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.to(device)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                encodings, **generation_params, pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        del encodings
        del generated_tokens
        torch.cuda.empty_cache()
        return generated_text

    # def _infer_text_openai(self, prompts):
    #     generation_params = {
    #         "temperature": 0.2,
    #         "max_tokens": 70,
    #         "top_p": 1.0,
    #         "frequency_penalty": 0.0,
    #         "presence_penalty": 0.0,
    #     }
    #     generated_texts = []
    #     for prompt in prompts:
    #         content = """
    #         Create an activity name that summarizes the following information from user interaction. Use the information given to get the context and deduce a meaningful activity.
    #         An example is: {{
    #             "Context (App Name)": "Todo App",
    #             "Context (App Category)": "Task Management",
    #             "Component Class": "Button",
    #             "Inner Text": " Delete",
    #             "Surrounding Text": "Buy milk; Delete;"
    #         }}
    #         Activity Name: Delete the task 'Buy milk'
    #         __________
    #         {prompt_text}
    #         __________
    #         Activity Name:
    #         """.format(
    #             prompt_text=json.dumps(prompt)
    #         )
    #         completion = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are Activity Name Generator for UI components. Answer as concisely as possible.",
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": content,
    #                 },
    #             ],
    #             **generation_params,
    #         )
    #         generated_text = completion["choices"][0]["message"]["content"]
    #         print(generated_text)
    #         generated_texts.append(generated_text)

    #     return generated_texts

    def _process_new_entries(
        self,
        new_entries,
        generated_text,
        resized_image,
    ):
        generated_activity_entries = []
        for index, new_entry in enumerate(new_entries):
            activity_name = ""
            if CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "basic":
                activity_name = self._determine_match(generated_text[index])
            elif CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "extended":
                activity_name = generated_text[index]
            if activity_name != "":
                new_entry["activity_name"] = activity_name
                new_entry["existing"] = "false"
                new_entry.pop("existing_activity_name", None)
                generated_activity_entries.append(new_entry)
                self.visualize_element(resized_image, new_entry["bbox"], activity_name)

                if CONFIG_PARSER.get(
                    "MAIN", "activity_generation_mode"
                ) == "basic" and CONFIG_PARSER.getboolean(
                    "MAIN", "save_activity_names"
                ):
                    self.db.insert_data(
                        new_entry["prompt_info"]["Context (App Name)"],
                        new_entry["prompt_info"]["Component Class"],
                        new_entry["prompt_info"]["Inner Text"],
                        activity_name,
                    )
                elif CONFIG_PARSER.get(
                    "MAIN", "activity_generation_mode"
                ) == "extended" and CONFIG_PARSER.getboolean(
                    "MAIN", "save_activity_names"
                ):
                    self.db.insert_data(
                        new_entry["prompt_info"]["Context (App Name)"],
                        new_entry["prompt_info"]["Context (App Category)"],
                        new_entry["prompt_info"]["Component Class"],
                        new_entry["prompt_info"]["Inner Text"],
                        new_entry["prompt_info"]["Surrounding Text"],
                        activity_name,
                    )
        return generated_activity_entries

    def _process_existing_entries(self, existing_entries, resized_image):
        generated_activity_entries = []
        for existing_entry in existing_entries:
            existing_entry["activity_name"] = existing_entry["existing_activity_name"]
            existing_entry.pop("existing_activity_name", None)
            existing_entry["existing"] = "true"
            generated_activity_entries.append(existing_entry)
            self.visualize_element(
                resized_image, existing_entry["bbox"], existing_entry["activity_name"]
            )
        return generated_activity_entries

    def is_long_text(self, text):
        words = text.split()
        return len(words) >= CONFIG_PARSER.getint("MAIN", "long_text_min_length")

    # def filter_out_irrelevant_compo(self, compos):
    # new_compos = []
    # TABU_LIST = ["Image", "Block", "Compo"]
    # for compo in compos:
    #     comp_class = str(compo["class"])
    #     if comp_class.lower() in TABU_LIST:
    #         continue
    #     if (
    #         comp_class == "Text"
    #         and compo["text_content"]
    #         and self.is_long_text(compo["text_content"])
    #     ):
    #         continue

    #     if (
    #         (
    #             comp_class == "Text"
    #             or comp_class == "Image"
    #             or comp_class == "Icon"
    #         )
    #         and "parent" in compo
    #         and (
    #             compos[compo["parent"]]["class"] == "Button"
    #             or compos[compo["parent"]]["class"] == "EditText"
    #         )
    #     ):
    #         continue
    #     new_compos.append(compo)

    # return new_compos

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

    # def get_all_text_from_id(self, compo_id, compos):
    #     text_content = ""

    #     current_compo = compos[compo_id]
    #     if "children" in current_compo:
    #         for child_compo_id in current_compo["children"]:
    #             child_compo = compos[child_compo_id]
    #             if child_compo["class"] == "Text" and "text_content" in child_compo:
    #                 text_content += child_compo["text_content"] + "; "

    #     return text_content.strip()

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

    # def find_nearby_text(self, target_compo, compos, threshold_distance=200):
    #     nearby_text = []

    #     if target_compo is None:
    #         return nearby_text

    #     target_compo_center_x = (
    #         int(target_compo["position"]["x1"]) + int(target_compo["position"]["x2"])
    #     ) / 2
    #     target_compo_center_y = (
    #         int(target_compo["position"]["y1"]) + int(target_compo["position"]["y2"])
    #     ) / 2

    #     for compo in compos:
    #         if compo["class"] == "Text":
    #             print(compo["text_content"])
    #             text_center_x = (
    #                 int((compo["position"]["x1"]) + int(compo["position"]["x2"])) / 2
    #             )
    #             text_center_y = (
    #                 int((compo["position"]["y1"]) + int(compo["position"]["y2"])) / 2
    #             )

    #             # Calculate the distance between the centers
    #             distance = (
    #                 (target_compo_center_x - text_center_x) ** 2
    #                 + (target_compo_center_y - text_center_y) ** 2
    #             ) ** 0.5

    #             if distance <= threshold_distance:
    #                 nearby_text.append(compo["text_content"])
    #     print(nearby_text)
    #     return " ".join(nearby_text)

    def find_nearby_text(
        self,
        target_compo,
        compos,
        threshold_distance=CONFIG_PARSER.getint(
            "MAIN", "find_nearby_text_threshold_distance"
        ),
    ):
        nearby_text = []

        if target_compo is None:
            return nearby_text

        target_x1 = int(target_compo["position"]["x1"])
        target_y1 = int(target_compo["position"]["y1"])
        target_x2 = int(target_compo["position"]["x2"])
        target_y2 = int(target_compo["position"]["y2"])

        for compo in compos:
            if compo["class"] == "Text":
                text_x1 = int(compo["position"]["x1"])
                text_y1 = int(compo["position"]["y1"])
                text_x2 = int(compo["position"]["x2"])
                text_y2 = int(compo["position"]["y2"])

                if (
                    target_x1 - threshold_distance <= text_x2
                    and target_x2 + threshold_distance >= text_x1
                    and target_y1 - threshold_distance <= text_y2
                    and target_y2 + threshold_distance >= text_y1
                ):
                    nearby_text.append(compo["text_content"])

        return " ".join(nearby_text)

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

            if (
                (comp_class == "Text" or comp_class == "Image" or "Icon" in comp_class)
                and "parent" in compo
                and (
                    compos[compo["parent"]]["class"] == "Button"
                    or compos[compo["parent"]]["class"] == "EditText"
                )
            ):
                continue

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
                        elif "Icon" in child["class"]:
                            inner_text += " " + child["class"]

            # Check surrounding text
            surrounding_text = ""
            block_parent_id = self._has_ancestor_block_or_compo(compo["id"], compos)
            if block_parent_id:
                surrounding_text = self.get_all_text_from_id(block_parent_id, compos)

            # Check Icons
            if "Icon" in comp_class:
                inner_text = comp_class.split("-")[-1]
                comp_class = "Icon"

            # Check Checkbox
            if "Checkbox" in comp_class:
                inner_text = "Uncheck checkbox"
                if "checkbox_state" in compo and compo["checkbox_state"] == "checked":
                    inner_text = "Check checkbox"
                if CONFIG_PARSER.getboolean("MAIN", "find_nearby_text"):
                    inner_text += self.find_nearby_text(compo, compos)

            if CONFIG_PARSER.getboolean("MAIN", "find_nearby_text"):
                if comp_class == "EditText" and inner_text == "":
                    inner_text = self.find_nearby_text(compo, compos)

                surrounding_text += self.find_nearby_text(compo, compos)

            # if "checkbox_state" in compo:
            #     inner_text = compo["checkbox_state"]

            # if "Checkbox" in comp_class:
            #     if "checkbox_state" in compo and compo["checkbox_state"] == "checked":
            #         comp_class += " - checked"
            #     else:
            #         comp_class += " - unchecked"

            if CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "basic":
                prompt_info = {
                    "Context (App Name)": app_name,
                    "Component Class": comp_class,
                    "Inner Text": inner_text,
                }
            elif CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "extended":
                prompt_info = {
                    "Context (App Name)": app_name,
                    "Context (App Category)": category,
                    "Component Class": comp_class,
                    "Inner Text": inner_text,
                    "Surrounding Text": surrounding_text,
                }

            existing_activity_name = None
            if "activity_name" in compo:
                existing_activity_name = compo["activity_name"]
            elif CONFIG_PARSER.getboolean("MAIN", "save_activity_names"):
                if CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "basic":
                    existing_activity_name = self.db.check_entry_exists(
                        app_name, comp_class, inner_text
                    )
                elif (
                    CONFIG_PARSER.get("MAIN", "activity_generation_mode") == "extended"
                ):
                    existing_activity_name = self.db.check_entry_exists(
                        app_name, category, comp_class, inner_text, surrounding_text
                    )

            entries.append(
                {
                    "bbox": {
                        "x1": compo["position"]["x1"],
                        "y1": compo["position"]["y1"],
                        "x2": compo["position"]["x2"],
                        "y2": compo["position"]["y2"],
                    },
                    "prompt_info": prompt_info,
                    "is_vism": compo["is_vism"],
                    "existing_activity_name": existing_activity_name,
                }
            )

        return entries

    def _determine_match(self, generated_text):
        activity_name = ""
        match = re.search(r"Activity Name:\s*(.+?)\\n", generated_text)
        if match:
            activity_name = match.group(1)
        return activity_name

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
