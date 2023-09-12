import numpy as np


class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.width = x2 - x1
        self.height = y2 - y1
        self.box_area = self.width * self.height

    def put_bbox(self):
        return self.x1, self.y1, self.x2, self.y2

    def bbox_cal_area(self):
        self.box_area = self.width * self.height
        return self.box_area

    def bbox_relation(self, bbox_b):
        """
        :return: -1 : a in b
                 0  : a, b are not intersected
                 1  : b in a
                 2  : a, b are identical or intersected
        """
        x1_a, y1_a, x2_a, y2_a = self.put_bbox()
        x1_b, y1_b, x2_b, y2_b = bbox_b.put_bbox()

        # if a is in b
        if x1_a > x1_b and y1_a > y1_b and x2_a < x2_b and y2_a < y2_b:
            return -1
        # if b is in a
        elif x1_a < x1_b and y1_a < y1_b and x2_a > x2_b and y2_a > y2_b:
            return 1
        # a and b are non-intersect
        elif (x1_a > x2_b or y1_a > y2_b) or (x1_b > x2_a or y1_b > y2_a):
            return 0
        # intersection
        else:
            return 2

    def bbox_relation_nms(self, bbox_b, bias=(0, 0)):
        """
         Calculate the relation between two rectangles by nms
        :return: -1 : a in b
          0  : a, b are not intersected
          1  : b in a
          2  : a, b are intersected
        """
        x1_a, y1_a, x2_a, y2_a = self.put_bbox()
        x1_b, y1_b, x2_b, y2_b = bbox_b.put_bbox()

        bias_col, bias_row = bias
        # get the intersected area
        x1_s = max(x1_a - bias_col, x1_b - bias_col)
        y1_s = max(y1_a - bias_row, y1_b - bias_row)
        x2_s = min(x2_a + bias_col, x2_b + bias_col)
        y2_s = min(y2_a + bias_row, y2_b + bias_row)
        w = np.maximum(0, x2_s - x1_s)
        h = np.maximum(0, y2_s - y1_s)
        inter = w * h
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        iou = inter / (area_a + area_b - inter)
        ioa = inter / self.box_area
        iob = inter / bbox_b.box_area

        if iou == 0 and ioa == 0 and iob == 0:
            return 0

        # import lib_ip.ip_preprocessing as pre
        # org_iou, _ = pre.read_img('uied/data/input/7.jpg', 800)
        # print(iou, ioa, iob)
        # board = draw.draw_bounding_box(org_iou, [self], color=(255,0,0))
        # draw.draw_bounding_box(board, [bbox_b], color=(0,255,0), show=True)

        # contained by b
        if ioa >= 1:
            return -1
        # contains b
        if iob >= 1:
            return 1
        # not intersected with each other
        # intersected
        if iou >= 0.02 or iob > 0.2 or ioa > 0.2:
            return 2
        # if iou == 0:
        # print('ioa:%.5f; iob:%.5f; iou:%.5f' % (ioa, iob, iou))
        return 0

    def bbox_cvt_relative_position(self, x1_base, y1_base):
        """
        Convert to relative position based on base coordinator
        """
        self.x1 += x1_base
        self.x2 += x1_base
        self.y1 += y1_base
        self.y2 += y1_base

    def bbox_merge(self, bbox_b):
        """
        Merge two intersected bboxes
        """
        x1_a, y1_a, x2_a, y2_a = self.put_bbox()
        x1_b, y1_b, x2_b, y2_b = bbox_b.put_bbox()
        x1 = min(x1_a, x1_b)
        x2 = max(x2_a, x2_b)
        y1 = min(y1_a, y1_b)
        y2 = max(y2_a, y2_b)
        new_bbox = Bbox(x1, y1, x2, y2)
        return new_bbox

    def bbox_padding(self, image_shape, pad):
        row, col = image_shape[:2]
        self.x1 = max(self.x1 - pad, 0)
        self.x2 = min(self.x2 + pad, col)
        self.y1 = max(self.y1 - pad, 0)
        self.y2 = min(self.y2 + pad, row)
