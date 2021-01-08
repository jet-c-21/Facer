"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/9
"""
# coding: utf-8
from typing import Union

import dlib
from numpy import ndarray

from .cls.captured_face import CapturedFace
from ..ult.read_img import get_img_arr


class FaceCapturer:
    def __init__(self, kernel='dlib'):
        self.kernel = kernel
        self.detector = None
        self.detector_loaded = False

    def _dlib_worker(self, img: ndarray) -> Union[CapturedFace, None]:
        try:
            detect_result = self.detector(img, 1)
        except Exception as e:
            msg = f"Failed to detect faces via dlib.get_frontal_face_detector(). Error: {e}"
            print(msg)
            return CapturedFace(exc=e)

        if len(detect_result):
            return CapturedFace(True, len(detect_result), detect_result, img, 'dlib')

        else:
            return CapturedFace(False, img=img, detector_type='dlib')

    def load_detector(self):
        if self.kernel is 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        self.detector_loaded = True

    def capture(self, img: Union[str, ndarray]) -> Union[CapturedFace, None]:
        img = get_img_arr(img)
        if self.kernel is 'dlib':
            return self._dlib_worker(img)
