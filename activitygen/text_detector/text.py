import cv2
import numpy as np


class Text:
    def __init__(self, id, content, location):
        self.id = id
        self.content = content
        self.location = location

        self.width = self.location["x2"] - self.location["x1"]
        self.height = self.location["y2"] - self.location["y1"]
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)

    def is_justified(self, ele_b, direction="h", max_bias_justify=4):
        l_a = self.location
        l_b = ele_b.location

        if direction == "v":
            if (
                abs(l_a["x1"] - l_b["x1"]) < max_bias_justify
                and abs(l_a["x2"] - l_b["x2"]) < max_bias_justify
            ):
                return True
            return False
        elif direction == "h":
            if (
                abs(l_a["y1"] - l_b["y1"]) < max_bias_justify
                and abs(l_a["y2"] - l_b["y2"]) < max_bias_justify
            ):
                return True
            return False

    def is_on_same_line(self, text_b, direction="h", bias_gap=4, bias_justify=4):
        l_a = self.location
        l_b = text_b.location

        if direction == "v":
            if self.is_justified(text_b, direction="v", max_bias_justify=bias_justify):
                if (
                    abs(l_a["y2"] - l_b["y1"]) < bias_gap
                    or abs(l_a["y1"] - l_b["y2"]) < bias_gap
                ):
                    return True
            return False
        elif direction == "h":
            if self.is_justified(text_b, direction="h", max_bias_justify=bias_justify):
                if (
                    abs(l_a["x2"] - l_b["x1"]) < bias_gap
                    or abs(l_a["x1"] - l_b["x2"]) < bias_gap
                ):
                    return True
            return False

    def is_intersected(self, text_b, bias):
        l_a = self.location
        l_b = text_b.location
        x1_in = max(l_a["x1"], l_b["x1"]) + bias
        y1_in = max(l_a["y1"], l_b["y1"]) + bias
        x2_in = min(l_a["x2"], l_b["x2"])
        y2_in = min(l_a["y2"], l_b["y2"])

        w_in = max(0, x2_in - x1_in)
        h_in = max(0, y2_in - y1_in)
        area_in = w_in * h_in
        if area_in > 0:
            return True

    def merge_text(self, text_b):
        text_a = self
        y1 = min(text_a.location["y1"], text_b.location["y1"])
        x1 = min(text_a.location["x1"], text_b.location["x1"])
        x2 = max(text_a.location["x2"], text_b.location["x2"])
        y2 = max(text_a.location["y2"], text_b.location["y2"])
        self.location = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        self.width = self.location["x2"] - self.location["x1"]
        self.height = self.location["y2"] - self.location["y1"]
        self.area = self.width * self.height

        x1_element = text_a
        x2_element = text_b
        if text_a.location["x1"] > text_b.location["x1"]:
            x1_element = text_b
            x2_element = text_a
        self.content = x1_element.content + " " + x2_element.content
        self.word_width = self.width / len(self.content)

    def shrink_bound(self, binary_map):
        bin_clip = binary_map[
            self.location["y1"] : self.location["y2"],
            self.location["x1"] : self.location["x2"],
        ]
        height, width = np.shape(bin_clip)

        shrink_y1 = 0
        shrink_y2 = 0
        for i in range(height):
            if shrink_y1 == 0:
                if sum(bin_clip[i]) == 0:
                    shrink_y1 = 1
                else:
                    shrink_y1 = -1
            elif shrink_y1 == 1:
                if sum(bin_clip[i]) != 0:
                    self.location["y1"] += i
                    shrink_y1 = -1
            if shrink_y2 == 0:
                if sum(bin_clip[height - i - 1]) == 0:
                    shrink_y2 = 1
                else:
                    shrink_y2 = -1
            elif shrink_y2 == 1:
                if sum(bin_clip[height - i - 1]) != 0:
                    self.location["y2"] -= i
                    shrink_y2 = -1

            if shrink_y1 == -1 and shrink_y2 == -1:
                break

        shrink_x1 = 0
        shrink_x2 = 0
        for j in range(width):
            if shrink_x1 == 0:
                if sum(bin_clip[:, j]) == 0:
                    shrink_x1 = 1
                else:
                    shrink_x1 = -1
            elif shrink_x1 == 1:
                if sum(bin_clip[:, j]) != 0:
                    self.location["x1"] += j
                    shrink_x1 = -1
            if shrink_x2 == 0:
                if sum(bin_clip[:, width - j - 1]) == 0:
                    shrink_x2 = 1
                else:
                    shrink_x2 = -1
            elif shrink_x2 == 1:
                if sum(bin_clip[:, width - j - 1]) != 0:
                    self.location["x2"] -= j
                    shrink_x2 = -1

            if shrink_x1 == -1 and shrink_x2 == -1:
                break

        self.width = self.location["x2"] - self.location["x1"]
        self.height = self.location["y2"] - self.location["y1"]
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)

    def visualize_element(self, img, color=(0, 0, 255), line=1, show=False):
        loc = self.location
        cv2.rectangle(img, (loc["x1"], loc["y1"]), (loc["x2"], loc["y2"]), color, line)
        if show:
            print(self.content)
            cv2.imshow("text", img)
            cv2.waitKey()
            cv2.destroyWindow("text")
