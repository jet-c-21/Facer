"""
author: Jet C.
GitHub: https://github.com/jet-chien
Create Date: 2021/1/9
"""
# coding: utf-8
from typing import Union

import face_recognition
import numpy
from imutils.paths import list_images
from numpy import ndarray
from tqdm import tqdm

from ..ult.data_store import save_as_pkl
from ..ult.read_img import get_img_arr


class AGFaceRecog:
    @staticmethod
    def get_face_encode(img: Union[str, ndarray]) -> Union[ndarray, None]:
        """
        :param img: str |  ndarray
        :return: ndarray (128)
        """
        img = get_img_arr(img)
        try:
            return face_recognition.face_encodings(img)[0]
        except Exception as e:
            msg = f"Failed to get face encodings. Error: {e}"
            print(msg)

    @staticmethod
    def data_to_fe_pkl(img_dir_path: str, save_path: str):
        result = list()
        image_ls = list(list_images(img_dir_path))
        for img_path in tqdm(image_ls):
            face_encode = AGFaceRecog.get_face_encode(img_path)
            if face_encode is not None:
                result.append(face_encode)

        msg = f"Data Length : {len(result)}"
        print(msg)
        save_as_pkl(result, save_path)

    @staticmethod
    def compare_faces(member_encodings: list, test_encoding: ndarray, tolerance=0.4) -> [bool]:
        return face_recognition.compare_faces(member_encodings, test_encoding, tolerance)

    @staticmethod
    def get_similarity(binary_ls: list) -> Union[float, ndarray]:
        return numpy.mean(binary_ls)

    @staticmethod
    def verify_member(member_encodings: list, test_encoding: ndarray, tolerance=0.4, threshold=0.6) -> (bool, float):
        res = AGFaceRecog.compare_faces(member_encodings, test_encoding, tolerance)
        similarity = AGFaceRecog.get_similarity(res)
        print(similarity)
        if similarity >= threshold:
            result = True
        else:
            result = False

        return result, similarity
