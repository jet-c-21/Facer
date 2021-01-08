"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
# coding: utf-8
from typing import Union

from numpy import ndarray

from .face_rotator import FaceRotator
from ..Detect.face_capturer import FaceCapturer
from ..Detect.lmk_scanner import LMKScanner


def get_rotated_face(img: Union[str, ndarray], face_capturer: FaceCapturer,
                         lmk_scanner: LMKScanner) -> Union[ndarray, None]:
    return FaceRotator.get_rotated_face(img, face_capturer, lmk_scanner)
