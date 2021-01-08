"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
# coding: utf-8
from typing import Union

import numpy as np

from .face_rotator import FaceRotator
from ..ult.read_img import get_img_arr


def get_rotated_face(img: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
    img = get_img_arr(img)
    return FaceRotator.get_rotated_face(img)
