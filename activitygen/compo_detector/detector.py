import os
from os.path import join as pjoin
import time

from activitygen.config.parameter_config import CONFIG_PARSER

from .processor import ImageProcessor
from .drawer import ImageDrawer
from . import detection_utils as det
from . import file_utils as file
from . import component as Compo

drawer = ImageDrawer()
processor = ImageProcessor()


class CompoDetector:
    def __init__(self, output_root):
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)

    def detect_compos(
        self, input_img_path, resize_by_height=800, show=False, wai_key=0
    ):
        start = time.perf_counter()
        name = (
            input_img_path.split("/")[-1][:-4]
            if "/" in input_img_path
            else input_img_path.split("\\")[-1][:-4]
        )
        ip_root = file.build_directory(pjoin(self.output_root, "compo"))

        # *** Step 1 *** pre-processing: read img -> get binary map
        org, grey = processor.read_img(input_img_path, resize_by_height)
        binary = processor.binarization(
            org, grad_min=CONFIG_PARSER.getfloat("MAIN", "min_grad")
        )

        # *** Step 2 *** element detection
        det.rm_line(binary, show=show, wait_key=wai_key)
        uicompos = det.component_detection(
            binary, min_obj_area=CONFIG_PARSER.getfloat("MAIN", "min_ele_area")
        )

        # *** Step 3 *** results refinement
        uicompos = det.compo_filter(
            uicompos,
            min_area=CONFIG_PARSER.getfloat("MAIN", "min_ele_area"),
            img_shape=binary.shape,
        )
        uicompos = det.merge_intersected_compos(uicompos)
        det.compo_block_recognition(binary, uicompos)
        if CONFIG_PARSER.getboolean("MAIN", "merge_contained_ele"):
            uicompos = det.rm_contained_compos_not_in_block(uicompos)
        Compo.compos_update(uicompos, org.shape)
        Compo.compos_containment(uicompos)

        # *** Step 4 ** nesting inspection: check if big compos have nesting element
        if CONFIG_PARSER.getboolean("MAIN", "activate_nesting_inspection"):
            uicompos += self._nesting_inspection(
                org,
                grey,
                uicompos,
                ffl_block=CONFIG_PARSER.getfloat("MAIN", "ffl_block"),
            )
        Compo.compos_update(uicompos, org.shape)
        drawer.draw_bounding_box(
            org,
            uicompos,
            show=show,
            name="merged compo",
            write_path=pjoin(ip_root, name + ".jpg"),
            wait_key=wai_key,
        )

        # *** Step 5 *** image inspection: recognize image -> remove noise in image -> binarize with larger threshold and reverse -> rectangular compo detection
        # if classifier is not None:
        #     classifier['Image'].predict(seg.clipping(org, uicompos), uicompos)
        #     draw.draw_bounding_box_class(org, uicompos, show=show)
        #     uicompos = det.rm_noise_in_large_img(uicompos, org)
        #     draw.draw_bounding_box_class(org, uicompos, show=show)
        #     det.detect_compos_in_img(uicompos, binary_org, org)
        #     draw.draw_bounding_box(org, uicompos, show=show)
        # if classifier is not None:
        #     classifier['Noise'].predict(seg.clipping(org, uicompos), uicompos)
        #     draw.draw_bounding_box_class(org, uicompos, show=show)
        #     uicompos = det.rm_noise_compos(uicompos)

        # *** Step 6 *** element classification: all category classification
        # if classifier is not None:
        #     classifier['Elements'].predict([compo.compo_clipping(org) for compo in uicompos], uicompos)
        #     draw.draw_bounding_box_class(org, uicompos, show=show, name='cls', write_path=pjoin(ip_root, 'result.jpg'))
        #     draw.draw_bounding_box_class(org, uicompos, write_path=pjoin(self.output_root, 'result.jpg'))

        # *** Step 7 *** save detection result
        Compo.compos_update(uicompos, org.shape)
        file.save_corners_json(pjoin(ip_root, name + ".json"), uicompos)
        if CONFIG_PARSER.getboolean("STEPS", "log_inbetween_outputs"):
            print(
                "[Compo Detection Completed in %.3f s] Input: %s Output: %s"
                % (
                    time.perf_counter() - start,
                    input_img_path,
                    pjoin(ip_root, name + ".json"),
                )
            )

    def _nesting_inspection(self, org, grey, compos, ffl_block):
        """
        Inspect all big compos through block division by flood-fill
        :param ffl_block: gradient threshold for flood-fill
        :return: nesting compos
        """
        nesting_compos = []
        for i, compo in enumerate(compos):
            if compo.height > 50:
                replace = False
                clip_grey = compo.compo_clipping(grey)
                n_compos = det.nested_components_detection(
                    clip_grey, org, grad_thresh=ffl_block, show=False
                )
                Compo.cvt_compos_relative_pos(n_compos, compo.bbox.x1, compo.bbox.y1)

                for n_compo in n_compos:
                    if n_compo.redundant:
                        compos[i] = n_compo
                        replace = True
                        break
                if not replace:
                    nesting_compos += n_compos
        return nesting_compos
