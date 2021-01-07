"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/8
"""
# coding: utf-8
import dlib
import numpy as np

from Facer.Crop.pure_cropper import PureCropper


def crop_face(img: np.ndarray, face_block: dlib.rectangle, margin=None):
    return PureCropper.get_cropped_face(img, face_block, margin)
