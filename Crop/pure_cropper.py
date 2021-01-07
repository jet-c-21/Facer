"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/8
"""
# coding: utf-8
import cv2
import dlib
import numpy as np

from Facer.Detect.capturer import capture_face

class PureCropper:
    MARGIN = 0.5

    @staticmethod
    def get_proper_margin(w: int, h: int, lt_x: int, lt_y: int, margin: float) -> float:
        w_expand = int(w * margin)
        h_expand = int(h * margin)

        new_lt_x = lt_x - w_expand
        new_lt_y = lt_y - h_expand

        if new_lt_x > 0 and new_lt_y > 0:
            delta_w = abs(new_lt_x - lt_x)
            delta_h = abs(new_lt_y - lt_y)
            proper_margin = max(delta_w / w, delta_h / h)

        else:
            if new_lt_x < 0:
                new_lt_x = 0

            if new_lt_y < 0:
                new_lt_y = 0

            delta_w = abs(new_lt_x - lt_x)
            delta_h = abs(new_lt_y - lt_y)
            proper_margin = min(delta_w / w, delta_h / h)

        return int(proper_margin * 100) / 100

    @staticmethod
    def get_cropped_face(img: np.ndarray, face_block: dlib.rectangle, margin=None) -> np.ndarray:
        if margin is None:
            margin = PureCropper.MARGIN
        else:
            margin = margin

        w = face_block.width()
        h = face_block.height()
        lt_x = face_block.left()
        lt_y = face_block.top()

        margin = PureCropper.get_proper_margin(w, h, lt_x, lt_y, margin)

        w_expand = int(w * margin)
        h_expand = int(h * margin)
        new_lt_x = lt_x - w_expand
        new_lt_y = lt_y - h_expand
        new_w = w + (2 * w_expand)
        new_h = h + (2 * h_expand)

        return img[new_lt_y:new_lt_y + new_h, new_lt_x:new_lt_x + new_w]

    @staticmethod
    def get_smart_cropped_face(img: np.ndarray, face_block: dlib.rectangle):
        for i in range(10, -1, -1):
            margin = i / 10
            cropped = PureCropper.get_cropped_face(img, face_block, margin)
            h, w, _ = cropped.shape
            if h > 0 and w > 0:
                capt_faces = capture_face(cropped)
                if capt_faces.face_count == 1:
                    return cropped
