"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
from typing import Union

# coding: utf-8
import dlib
import numpy as np

from Facer.Detect.cls.captured_face import CapturedFace
from Facer.ult.read_img import get_img_arr

dlib_detector = dlib.get_frontal_face_detector()


def capture_face(img: Union[str, np.ndarray], detector='dlib') -> CapturedFace:
    img = get_img_arr(img)

    if detector is 'dlib':
        try:
            detect_result = dlib_detector(img, 1)
        except Exception as e:
            msg = f"Failed to detect faces via dlib.get_frontal_face_detector(). Error: {e}"
            print(msg)
            return CapturedFace(exc=e)

        if len(detect_result):
            return CapturedFace(True, len(detect_result), detect_result, 'dlib')

        else:
            return CapturedFace(False, detector_type='dlib')
