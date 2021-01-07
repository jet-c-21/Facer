"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/7
"""
# coding: utf-8
import dlib


class CapturedFace:
    def __init__(self, has_face=None, face_count=None, face_ls=None, detector_type=None, exc=None):
        """

        :param has_face: bool
        :param face_count: int
        :param face_ls: list, [face_block (dlib.rectangle), ...]
        :param detector_type: str
        :param exc: str
        """
        if has_face is None:
            self.has_face = False
        else:
            self.has_face = has_face

        if face_count is None:
            self.face_count = 0
        else:
            self.face_count = face_count

        if face_ls is None:
            self.face_ls = list()
        else:
            self.face_list = face_ls

        if detector_type is None:
            self.detector_type = 'unknown'
        else:
            self.detector_type = detector_type

        if exc is None:
            self.exc = None
        else:
            self.exc = exc

    def __str__(self):
        if self.has_face:
            text = f"Has Face: {self.has_face}\n" \
                   f"Face Count: {self.face_count}\n" \
                   f"Detector Type: {self.detector_type}\n" \
                   f"Face List:\n"

            face_ls_str = ''
            if self.face_count <= 3:
                for i, f in enumerate(self.face_list, start=1):
                    face_ls_str += f"{' ' * 4}Face-{i}: {str(f)}"

            else:
                for i, f in enumerate(self.face_list, start=1):
                    face_ls_str += f"{' ' * 4}Face-{i}: {str(f)}\n"

            text += face_ls_str

            return text.strip()

        else:
            text = f"Has Face: {self.has_face}\n" \
                   f"Face Count: {self.face_count}\n" \
                   f"Detector Type: {self.detector_type}\n" \
                   f"Exception: {self.exc}"
            return text.strip()

    def __getitem__(self, item) -> dlib.rectangle:
        return self.face_list[item]
