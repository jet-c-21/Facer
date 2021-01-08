"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
# coding: utf-8
from typing import Union

import numpy as np

from .cls.captured_face import CapturedFace
from .face_capturer import FaceCapturer


def capture_face(img: Union[str, np.ndarray], face_capturer: FaceCapturer) -> CapturedFace:
    return face_capturer.capture(img)
