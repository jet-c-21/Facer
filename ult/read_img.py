"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
# coding: utf-8
import os
from typing import Union

import cv2
import numpy as np
from imutils.paths import list_images


def get_img_ls(root_dir_path: str) -> list:
    if not os.path.exists(root_dir_path):
        msg = f"[FACER][WARN] - Input directory is not existed! Path: {root_dir_path}"
        print(msg)
        return list()

    return list(list_images(root_dir_path))


def get_img_arr(input_obj: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
    if isinstance(input_obj, np.ndarray):
        return input_obj

    elif isinstance(input_obj, str):
        if os.path.exists(input_obj):
            try:
                return cv2.imread(input_obj)
            except Exception as e:
                msg = f"[FACER] - Failed to read image from path : {input_obj}. Error: {e}"
                print(msg)
