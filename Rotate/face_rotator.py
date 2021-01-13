"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
from typing import Union

# coding: utf-8
import cv2
from numpy import ndarray
import numpy as np
from dlib import rectangle

from ..Detect.face_capturer import FaceCapturer
from ..Detect.lmk_scanner import LMKScanner


class FaceRotator:
    LEFT_EYE_LMK_POS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_LMK_POS = [42, 43, 44, 45, 46, 47]

    @staticmethod
    def get_angle_to_hrzl(p1: tuple, p2: tuple):
        deltaY = p1[1] - p2[1]
        deltaX = p1[0] - p2[0]
        return np.arctan(deltaY / deltaX) * 180 / np.pi

    @staticmethod
    def rect_to_tuple(rect):
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        return left, top, right, bottom

    @staticmethod
    def extract_eye(shape, eye_indices):
        points = map(lambda i: shape.part(i), eye_indices)
        return list(points)

    @staticmethod
    def extract_eye_center(shape, eye_indices):
        points = FaceRotator.extract_eye(shape, eye_indices)
        xs = map(lambda p: p.x, points)
        ys = map(lambda p: p.y, points)
        return sum(xs) // 6, sum(ys) // 6

    @staticmethod
    def get_left_eye_center(land_marks):
        return FaceRotator.extract_eye_center(land_marks, FaceRotator.LEFT_EYE_LMK_POS)

    @staticmethod
    def get_right_eye_center(land_marks):
        return FaceRotator.extract_eye_center(land_marks, FaceRotator.RIGHT_EYE_LMK_POS)

    @staticmethod
    def get_rotated_img(image: ndarray, angle: float, center=None, scale=1.0) -> ndarray:
        h, w = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)

        # rotate
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @staticmethod
    def get_rotated_face(img: Union[str, ndarray], face_capturer: FaceCapturer,
                         lmk_scanner: LMKScanner) -> Union[ndarray, None]:
        capt_face = face_capturer.capture(img)

        if capt_face.face_count != 1:
            msg = f"The count of faces in the raw image is not equal to 1"
            # print(msg)
            return

        img = capt_face.img
        face_block = capt_face[0]
        return FaceRotator.get_rotated_face_with_fb(img, face_block, lmk_scanner)

    @staticmethod
    def get_rotated_face_with_fb(img: ndarray, face_block: rectangle,
                                 lmk_scanner: LMKScanner) -> Union[ndarray, None]:

        landmarks = lmk_scanner.scan(img, face_block)
        if landmarks is None:
            msg = f"Failed to get landmarks from image"
            print(msg)
            return

        left_eye_coord = FaceRotator.get_left_eye_center(landmarks)
        right_eye_coord = FaceRotator.get_right_eye_center(landmarks)
        angle_to_hrzl = FaceRotator.get_angle_to_hrzl(left_eye_coord, right_eye_coord)
        return FaceRotator.get_rotated_img(img, angle_to_hrzl)
