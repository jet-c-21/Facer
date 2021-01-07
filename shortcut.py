"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/8
"""
# coding: utf-8
from typing import Union

import numpy as np

from Facer.Crop.cropper import crop_face
from Facer.Detect.capturer import capture_face
from Facer.Rotate.rotator import get_rotated_face
from Facer.ult.read_img import get_img_arr


def get_face_grid_from_portrait(img: Union[str, np.ndarray], margin=None) -> Union[np.ndarray, None]:
    img = get_img_arr(img)

    # rotate image
    rotated_face = get_rotated_face(img)
    if rotated_face is None:
        return

    # get new face block from the rotated image
    capt_face = capture_face(rotated_face)
    if capt_face.face_count != 1:
        msg = 'Failed to get face grid from image, the face count of the rotated image != 1'
        print(msg)
        return

    face_block = capt_face[0]
    return crop_face(rotated_face, face_block, margin)
