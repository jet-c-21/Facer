"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
from typing import Union

# coding: utf-8
import cv2
import dlib
import numpy as np

from Facer.Detect.capturer import capture_face
from Facer.ult.model_api import ModelAPI


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
    def get_landmarks(img: np.ndarray, face_block: dlib.rectangle) -> Union[dlib.full_object_detection, None]:
        lmk_detector = ModelAPI.get('dlib-lmk68')
        try:
            return lmk_detector(img, face_block)
        except Exception as e:
            print(f"failed to detect landmarks from face.  Error: {e}")

    @staticmethod
    def get_rotated_img(image: np.ndarray, angle: float, center=None, scale=1.0) -> np.ndarray:
        h, w = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)

        # rotate
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @staticmethod
    def get_rotated_face(img: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
        """

        :param img: str | np.ndarray
        :return:
        """
        capt_face = capture_face(img)
        if capt_face.face_count != 1:
            msg = f"The count of faces in the image is not 1"
            print(msg)
            return

        face_block = capt_face[0]
        return FaceRotator.get_rotated_face_with_fb(img, face_block)

    @staticmethod
    def get_rotated_face_with_fb(img: np.ndarray, face_block: dlib.rectangle) -> Union[np.ndarray, None]:
        """

        :param img: np.ndarray
        :param face_block: dlib.rectangle
        :return:
        """
        landmarks = FaceRotator.get_landmarks(img, face_block)
        if landmarks is None:
            msg = f"Failed to get landmarks from image"
            print(msg)
            return

        left_eye_coord = FaceRotator.get_left_eye_center(landmarks)
        right_eye_coord = FaceRotator.get_right_eye_center(landmarks)
        angle_to_hrzl = FaceRotator.get_angle_to_hrzl(left_eye_coord, right_eye_coord)
        return FaceRotator.get_rotated_img(img, angle_to_hrzl)
