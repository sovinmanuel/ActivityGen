import numpy as np
import cv2


class Element:
    def __init__(self, id, corner, category, text_content=None):
        self.id = id
        self.category = category
        self.x1, self.y1, self.x2, self.y2 = corner
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height

        self.text_content = text_content
        # self.text_color: str | None = None
        self.checkbox_state: str | None = None
        self.negative_template: bool = False
        self.is_custom: bool = False
        self.is_vism: bool = False
        self.activity_name: str | None = None
        self.parent_id = None
        self.children = []  # list of elements

    def init_bound(self):
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height

    def put_bbox(self):
        return self.x1, self.y1, self.x2, self.y2

    def wrap_info(self):
        info = {
            "id": self.id,
            "class": self.category,
            "height": self.height,
            "width": self.width,
            "position": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
        }
        if self.text_content is not None:
            info["text_content"] = self.text_content
        if len(self.children) > 0:
            info["children"] = [child.id for child in self.children]
        if self.parent_id is not None:
            info["parent"] = self.parent_id
        # if self.text_color is not None:
        #     info["text_color"] = self.text_color
        if self.checkbox_state is not None:
            info["checkbox_state"] = self.checkbox_state
        if self.negative_template is not None:
            info["negative_template"] = self.negative_template
        if self.is_custom is not None:
            info["is_custom"] = self.is_custom
        if self.is_vism is not None:
            info["is_vism"] = self.is_vism
        if self.activity_name is not None:
            info["activity_name"] = self.activity_name
        return info

    def resize(self, resize_ratio):
        self.x1 = int(self.x1 * resize_ratio)
        self.y1 = int(self.y1 * resize_ratio)
        self.x2 = int(self.x2 * resize_ratio)
        self.y2 = int(self.y2 * resize_ratio)
        self.init_bound()

    def element_merge(
        self, element_b, new_element=False, new_category=None, new_id=None
    ):
        x1_a, y1_a, x2_a, y2_a = self.put_bbox()
        x1_b, y1_b, x2_b, y2_b = element_b.put_bbox()
        new_corner = (
            min(x1_a, x1_b),
            min(y1_a, y1_b),
            max(x2_a, x2_b),
            max(y2_a, y2_b),
        )
        if element_b.text_content is not None:
            self.text_content = (
                element_b.text_content
                if self.text_content is None
                else self.text_content + "\n" + element_b.text_content
            )
        if new_element:
            return Element(new_id, new_corner, new_category)
        else:
            self.x1, self.y1, self.x2, self.y2 = new_corner
            self.init_bound()

    def calc_intersection_area(self, element_b, bias=(0, 0)):
        a = self.put_bbox()
        b = element_b.put_bbox()
        x1_s = max(a[0], b[0]) - bias[0]
        y1_s = max(a[1], b[1]) - bias[1]
        x2_s = min(a[2], b[2])
        y2_s = min(a[3], b[3])
        w = np.maximum(0, x2_s - x1_s)
        h = np.maximum(0, y2_s - y1_s)
        inter = w * h

        iou = inter / (self.area + element_b.area - inter)
        ioa = inter / self.area
        iob = inter / element_b.area

        return inter, iou, ioa, iob

    def element_relation(self, element_b, bias=(0, 0)):
        """
        @bias: (horizontal bias, vertical bias)
        :return: -1 : a in b
                 0  : a, b are not intersected
                 1  : b in a
                 2  : a, b are identical or intersected
        """
        inter, iou, ioa, iob = self.calc_intersection_area(element_b, bias)

        # area of intersection is 0
        if ioa == 0:
            return 0
        # a in b
        if ioa >= 1:
            return -1
        # b in a
        if iob >= 1:
            return 1
        return 2

    def visualize_element(self, img, color=(0, 255, 0), line=1, text="[]", show=False):
        loc = self.put_bbox()
        x_1, y_1 = loc[:2]
        x_2, y_2 = loc[2:]
        img = cv2.rectangle(img, (x_1, y_1), (x_2, y_2), color, line)
        img = cv2.putText(
            img, text, (x_1, y_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (36, 255, 12), 2
        )
        if show:
            cv2.imshow("element", img)
            cv2.waitKey(0)
            cv2.destroyWindow("element")

    def compo_clipping(self, img, pad=5, show=False):
        (x1, y1, x2, y2) = self.put_bbox()
        x1 = max(x1 - pad, 0)
        x2 = min(x2 + pad, img.shape[1])
        y1 = max(y1 - pad, 0)
        y2 = min(y2 + pad, img.shape[0])
        clip = img[y1:y2, x1:x2]
        if show:
            cv2.imshow("clipping", clip)
            cv2.waitKey()
        return clip
