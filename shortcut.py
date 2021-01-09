"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/8
"""
# coding: utf-8
from typing import Union

import numpy as np

from .Crop.cropper import crop_face
from .Detect.capturer import capture_face
from .Detect.face_capturer import FaceCapturer
from .Detect.lmk_scanner import LMKScanner
from .Rotate.rotator import get_rotated_face


def get_face_grid_from_portrait(img: Union[str, np.ndarray], face_capturer: FaceCapturer,
                                lmk_scanner: LMKScanner, margin=0.2) -> Union[np.ndarray, None]:
    # rotate image
    rotated_face = get_rotated_face(img, face_capturer, lmk_scanner)

    if rotated_face is None:
        return

    # get new face block from the rotated image
    capt_face = capture_face(rotated_face, face_capturer)

    # print(ed - st)
    if capt_face.face_count != 1:
        msg = 'Failed to get face grid from image, the face count of the rotated image != 1'
        print(msg)
        return
    face_block = capt_face[0]

    # crop face grid
    cropped_face = crop_face(rotated_face, face_block, margin)

    return cropped_face
