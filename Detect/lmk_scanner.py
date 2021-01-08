"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/9
"""
# coding: utf-8
from numpy import ndarray
from typing import Union
from ..ult.model_api import ModelAPI
from dlib import rectangle, full_object_detection

class LMKScanner:
    def __init__(self, kernel='dlib'):
        self.kernel = kernel
        self.detector = None
        self.detector_loaded = False

    def load_detector(self):
        if self.kernel is 'dlib':
            self.detector = ModelAPI.get('dlib-lmk68')
        self.detector_loaded = True

    def scan(self, img: ndarray, face_block: rectangle) -> Union[full_object_detection, None]:
        try:
            return self.detector(img, face_block)
        except Exception as e:
            print(f"failed to detect landmarks from face.  Error: {e}")
