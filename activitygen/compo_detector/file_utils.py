import os
import pandas as pd
import json
from os.path import join as pjoin
import cv2
import numpy as np
from PIL import Image


def save_corners(file_path, corners, compo_name, clear=True):
    try:
        df = pd.read_csv(file_path, index_col=0)
    except:
        df = pd.DataFrame(
            columns=["component", "x_max", "x_min", "y_max", "y_min", "height", "width"]
        )

    if clear:
        df = df.drop(df.index)
    for corner in corners:
        (up_left, bottom_right) = corner
        c = {"component": compo_name}
        (c["y_min"], c["x_min"]) = up_left
        (c["y_max"], c["x_max"]) = bottom_right
        c["width"] = c["y_max"] - c["y_min"]
        c["height"] = c["x_max"] - c["x_min"]
        df = df.append(c)
    df.to_csv(file_path)


def save_corners_json(file_path, compos):
    img_shape = compos[0].image_shape
    output = {"img_shape": img_shape, "compos": []}
    f_out = open(file_path, "w")

    for compo in compos:
        c = {"id": compo.id, "class": compo.category}
        (
            c["x1"],
            c["y1"],
            c["x2"],
            c["y2"],
        ) = compo.put_bbox()
        c["width"] = compo.width
        c["height"] = compo.height
        output["compos"].append(c)

    json.dump(output, f_out, indent=4)


def save_clipping(org, output_root, corners, compo_classes, compo_index):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    pad = 2
    for i in range(len(corners)):
        compo = compo_classes[i]
        (up_left, bottom_right) = corners[i]
        (x1, y1) = up_left
        (x2, y2) = bottom_right
        x1 = max(x1 - pad, 0)
        x2 = min(x2 + pad, org.shape[1])
        y1 = max(y1 - pad, 0)
        y2 = min(y2 + pad, org.shape[0])

        # if component type already exists, index increase by 1, otherwise add this type
        compo_path = pjoin(output_root, compo)
        if compo_classes[i] not in compo_index:
            compo_index[compo_classes[i]] = 0
            if not os.path.exists(compo_path):
                os.mkdir(compo_path)
        else:
            compo_index[compo_classes[i]] += 1
        clip = org[y1:y2, x1:x2]
        cv2.imwrite(
            pjoin(compo_path, str(compo_index[compo_classes[i]]) + ".png"), clip
        )


def build_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def read_and_resize_image(input_img_path, resize_height=800):
    image = Image.open(input_img_path)
    image = image.convert("RGB")
    w_h_ratio = image.width / image.height
    resize_width = int(resize_height * w_h_ratio)
    resized_image = image.resize((resize_width, resize_height))
    return resized_image


def convert_from_cv2_to_pil(img):
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_pil_to_cv2(img):
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_image_shape(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image.shape


def draw_matches(input_img_path, matches, resize_height=800):
    image = read_and_resize_image(input_img_path, resize_height)
    image = convert_from_pil_to_cv2(image)
    for i, match in enumerate(matches):
        cv2.rectangle(
            image,
            (match["x1"], match["y1"]),
            (match["x2"], match["y2"]),
            (0, 255, 0),
            2,
        )
    cv2.imshow("Matches", image)
    cv2.waitKey(0)
