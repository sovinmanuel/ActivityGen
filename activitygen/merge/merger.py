from configparser import ConfigParser
import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import shutil
from activitygen.compo_detector.drawer import ImageDrawer
from activitygen.compo_classifier.classifier import ClassifierPipeline
from activitygen.config.parameter_config import CONFIG_PARSER
from .element import Element

drawer = ImageDrawer()


def show_elements(
    org_img, eles, show=False, win_name="element", wait_key=0, shown_resize=None, line=8
):
    # color_map = {'Text':(0, 0, 255), 'Compo':(0, 255, 0), 'Block':(0, 255, 0), 'Text Content':(255, 0, 255)}
    img = org_img.copy()
    for ele in eles:
        # color = color_map[ele.category]
        ele.visualize_element(img, (0, 0, 255), line, ele.category)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(win_name)
    return img_resize


def save_elements(output_file, elements, img_shape, raw_texts):
    info_obj = {"compos": [], "img_shape": img_shape, "raw_texts": raw_texts}
    for i, ele in enumerate(elements):
        c = ele.wrap_info()
        # c['id'] = i
        info_obj["compos"].append(c)
    json.dump(info_obj, open(output_file, "w"), indent=4)
    return info_obj


def reassign_ids(elements):
    for i, element in enumerate(elements):
        element.id = i


def refine_texts(texts, img_shape):
    refined_texts = []
    for text in texts:
        # remove potential noise
        if len(text.text_content) > 1 and text.height / img_shape[0] < 0.075:
            refined_texts.append(text)
    return refined_texts


def merge_text_line_to_paragraph(elements, max_line_gap=5):
    texts = []
    non_texts = []
    for ele in elements:
        if ele.category == "Text":
            texts.append(ele)
        else:
            non_texts.append(ele)

    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                inter_area, _, _, _ = text_a.calc_intersection_area(
                    text_b, bias=(0, max_line_gap)
                )
                if inter_area > 0:
                    text_b.element_merge(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return non_texts + texts


def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8):
    """
    1. remove compos contained in text
    2. remove compos containing text area that's too large
    3. store text in a compo if it's contained by the compo as the compo's text child element
    """
    elements = []
    contained_texts = []
    for compo in compos:
        is_valid = True
        text_area = 0
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(
                text, bias=intersection_bias
            )
            if inter > 0:
                # the non-text is contained in the text compo
                if ioa >= containment_ratio:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != "Block":
                    contained_texts.append(text)
        if is_valid and text_area / compo.area < containment_ratio:
            # for t in contained_texts:
            #     t.parent_id = compo.id
            # compo.children += contained_texts
            elements.append(compo)

    # elements += texts
    for text in texts:
        # if text not in contained_texts:
        elements.append(text)
    return elements


def refine_tables(elements, tables, intersection_bias=(2, 2), containment_ratio=0.8):
    """
    Omit all components that are in tables.
    """
    refined_elements = []
    for element in elements:
        is_valid = True
        for table in tables:
            inter, _, _, _ = element.calc_intersection_area(
                table, bias=intersection_bias
            )
            if inter > 0:
                is_valid = False
                break
        if is_valid:
            refined_elements.append(element)
    for table in tables:
        # if text not in contained_texts:
        refined_elements.append(table)
    return refined_elements


def refine_matches(elements, matches, intersection_bias=(2, 2), containment_ratio=0.1):
    refined_elements = []
    for ele in elements:
        is_valid = True
        for match in matches:
            inter, iou, ioa, iob = ele.calc_intersection_area(
                match, bias=intersection_bias
            )
            if inter > 0:
                if containment_ratio <= iou:
                    is_valid = False
                    break
        if is_valid:
            refined_elements.append(ele)

    for match in matches:
        if not match.negative_template:
            refined_elements.append(match)
    return refined_elements


def remove_negative_templates(elements):
    refined_elements = []
    for ele in elements:
        if not ele.negative_template:
            refined_elements.append(ele)
    return refined_elements


def only_allow_vism(elements):
    vism_elements = []
    for ele in elements:
        if ele.is_vism:
            vism_elements.append(ele)
    return vism_elements


def check_containment(elements):
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


def reset_containment(elements):
    for i in range(len(elements)):
        elements[i].children = []
        elements[i].parent_id = None


def remove_top_bar(elements, img_height):
    new_elements = []
    max_height = img_height * CONFIG_PARSER.getfloat(
        "MAIN", "remove_top_bar_ratio"
    )  # Param for remove bar, 0.04 for mobile, 0.1 for desktop
    for ele in elements:
        if ele.y1 < max_height:
            continue
        new_elements.append(ele)
    return new_elements


def remove_children_from_image(elements):
    new_elements = []
    for ele in elements:
        has_image_ancestor = False
        current_element = ele
        while current_element and current_element.parent_id is not None:
            parent = next(
                (e for e in elements if e.id == current_element.parent_id), None
            )
            if parent and parent.category == "Image":
                has_image_ancestor = True
                break
            current_element = parent

        if has_image_ancestor:
            continue

        new_elements.append(ele)

    return new_elements


def remove_children_from_button(elements):
    new_elements = []
    for ele in elements:
        has_btn_ancestor = False
        current_element = ele
        while current_element and current_element.parent_id is not None:
            parent = next(
                (e for e in elements if e.id == current_element.parent_id), None
            )
            if parent and parent.category == "Button":
                has_btn_ancestor = True
                break
            current_element = parent

        # print(ele.category)
        if has_btn_ancestor:
            if ele.category != "Text":
                # print(has_btn_ancestor)
                # print(ele.category)
                continue

        new_elements.append(ele)

    return new_elements


def remove_children_from_icon(elements):
    new_elements = []
    for ele in elements:
        has_icon_ancestor = False
        current_element = ele
        while current_element and current_element.parent_id is not None:
            parent = next(
                (e for e in elements if e.id == current_element.parent_id), None
            )
            if parent and "Icon" in parent.category:
                has_icon_ancestor = True
                break
            current_element = parent

        if has_icon_ancestor:
            continue

        new_elements.append(ele)

    return new_elements


def remove_bottom_bar(elements, img_height):
    new_elements = []
    for ele in elements:
        # parameters for 800-height GUI
        if ele.y1 > 750 and 20 <= ele.height <= 30 and 20 <= ele.width <= 30:
            continue
        new_elements.append(ele)
    return new_elements


# def remove_bottom_bar(elements, img_height):
#     new_elements = []
#     max_height = img_height * CONFIG_PARSER.getfloat(
#         "MAIN", "remove_bottom_bar_ratio"
#     )  # Param for remove bar, 0.04 for mobile, 0.1 for desktop
#     for ele in elements:
#         if ele.y2 > img_height - max_height:
#             continue
#         new_elements.append(ele)
#     return new_elements


def compos_clip_and_fill(clip_root, org, compos):
    def most_pix_around(pad=6, offset=2):
        """
        determine the filled background color according to the most surrounding pixel
        """
        up = y1 - pad if y1 - pad >= 0 else 0
        left = x1 - pad if x1 - pad >= 0 else 0
        bottom = y2 + pad if y2 + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = x2 + pad if x2 + pad < org.shape[1] - 1 else org.shape[1] - 1
        most = []
        for i in range(3):
            val = np.concatenate(
                (
                    org[up : y1 - offset, left:right, i].flatten(),
                    org[y2 + offset : bottom, left:right, i].flatten(),
                    org[up:bottom, left : x1 - offset, i].flatten(),
                    org[up:bottom, x2 + offset : right, i].flatten(),
                )
            )
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo["class"]
        if cls == "Background":
            compo["path"] = pjoin(clip_root, "bkg.png")
            continue
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo["id"]) + ".jpg")
        compo["path"] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        position = compo["position"]
        x1, y1, x2, y2 = (
            position["x1"],
            position["y1"],
            position["x2"],
            position["y2"],
        )
        cv2.imwrite(c_path, org[y1:y2, x1:x2])
        # Fill up the background area
        cv2.rectangle(bkg, (x1, y1), (x2, y2), most_pix_around(), -1)
    cv2.imwrite(pjoin(clip_root, "bkg.png"), bkg)


# def get_text_color(text_img):
#     return "black"


def merge(
    img_path,
    compo_path,
    text_path,
    table_path=None,
    tm_path=None,
    fm_path=None,
    merge_root=None,
    classifier=None,
    show=False,
    wait_key=0,
):
    compo_json = json.load(open(compo_path, "r"))
    text_json = json.load(open(text_path, "r"))

    img = cv2.imread(img_path)
    img_resize = cv2.resize(
        img, (compo_json["img_shape"][1], compo_json["img_shape"][0])
    )
    img_resize_unchanged = img_resize.copy()

    # load text and non-text compo
    ele_id = 0
    compos = []
    for compo in compo_json["compos"]:
        element = Element(
            ele_id,
            (
                compo["x1"],
                compo["y1"],
                compo["x2"],
                compo["y2"],
            ),
            compo["class"],
        )
        compos.append(element)
        ele_id += 1
    texts = []
    for text in text_json["texts"]:
        element = Element(
            ele_id,
            (text["x1"], text["y1"], text["x2"], text["y2"]),
            "Text",
            text_content=text["content"],
        )
        # element.text_color = get_text_color(element.compo_clipping(img_resize.copy()))
        # print("element.text_color")
        # print(element.text_color)
        texts.append(element)
        ele_id += 1
    tables = []
    if table_path:
        table_json = json.load(open(table_path, "r"))
        for table in table_json["tables"]:
            element = Element(
                ele_id,
                (
                    table["x1"],
                    table["y1"],
                    table["x2"],
                    table["y2"],
                ),
                "Table",
            )
            element.is_custom = True
            tables.append(element)
            ele_id += 1
    tm_matches = []
    if tm_path:
        tm_json = json.load(open(tm_path, "r"))
        for tm_match in tm_json["matches"]:
            element = Element(
                ele_id,
                (
                    tm_match["x1"],
                    tm_match["y1"],
                    tm_match["x2"],
                    tm_match["y2"],
                ),
                tm_match["template_name"],
            )
            # print("tm_match[template_name]")
            # print(tm_match["template_name"])
            negative_template_val = tm_match["negative_template"].lower()
            negative_template = False
            if negative_template_val == "true":
                negative_template = True
            element.negative_template = negative_template
            element.is_custom = True
            if "activity_name" in tm_match:
                element.activity_name = tm_match["activity_name"]
            tm_matches.append(element)
            ele_id += 1
    fm_matches = []
    if fm_path:
        fm_json = json.load(open(fm_path, "r"))
        for fm_match in fm_json["matches"]:
            element = Element(
                ele_id,
                (
                    fm_match["x1"],
                    fm_match["y1"],
                    fm_match["x2"],
                    fm_match["y2"],
                ),
                fm_match["template_name"],
            )
            negative_template_val = fm_match["negative_template"].lower()
            negative_template = False
            if negative_template_val == "true":
                negative_template = True
            element.negative_template = negative_template
            element.is_custom = True
            if "activity_name" in fm_match:
                element.activity_name = fm_match["activity_name"]
            fm_matches.append(element)
            ele_id += 1
    if compo_json["img_shape"] != text_json["img_shape"]:
        resize_ratio = compo_json["img_shape"][0] / text_json["img_shape"][0]
        for text in texts:
            text.resize(resize_ratio)

    # check the original detected elements
    show_elements(
        img_resize,
        texts + compos,
        show=show,
        win_name="all elements before merging",
        wait_key=wait_key,
    )

    # refine elements
    texts = refine_texts(texts, compo_json["img_shape"])
    elements = refine_elements(compos, texts)
    if tables:
        elements = refine_tables(elements, tables)
    if tm_matches:
        elements = refine_matches(elements, tm_matches)
    if fm_matches:
        elements = refine_matches(elements, fm_matches)
    if CONFIG_PARSER.getboolean("MAIN", "remove_bar"):
        elements = remove_top_bar(elements, img_height=compo_json["img_shape"][0])
        elements = remove_bottom_bar(elements, img_height=compo_json["img_shape"][0])
    if CONFIG_PARSER.getboolean("MAIN", "merge_line_to_paragraph"):
        elements = merge_text_line_to_paragraph(
            elements, max_line_gap=CONFIG_PARSER.getfloat("MAIN", "max_line_gap")
        )
    reassign_ids(elements)
    check_containment(elements)

    # Classify components
    if classifier:
        classifier.predict(
            [elem.compo_clipping(img_resize.copy()) for elem in elements], elements
        )

    # elements = remove_children_from_image(elements)
    # elements = remove_children_from_button(elements)
    # elements = remove_children_from_icon(elements)
    remove_negative_templates(elements)
    if CONFIG_PARSER.getboolean("MAIN", "only_allow_vism"):
        elements = only_allow_vism(elements)
    reset_containment(elements)
    reassign_ids(elements)
    check_containment(elements)

    board = show_elements(
        img_resize,
        elements,
        show=show,
        win_name="elements after merging",
        wait_key=wait_key,
    )

    raw_texts = ""
    for element in elements:
        if element.category == "Text" and element.text_content:
            raw_texts += element.text_content + "; "
    raw_texts = raw_texts[:-2]
    # save all merged elements, clips and blank background
    name = img_path.replace("\\", "/").split("/")[-1][:-4]
    info_obj = save_elements(
        pjoin(merge_root, name + ".json"), elements, img_resize.shape, raw_texts
    )
    cv2.imwrite(pjoin(merge_root, name + ".jpg"), board)
    if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
        print(
            "[Merge Completed] Input: %s Output: %s"
            % (img_path, pjoin(merge_root, name + ".jpg"))
        )
    return img_resize.shape, img_resize_unchanged, info_obj
