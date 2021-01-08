"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/1/9
"""
# coding: utf-8
import cv2
from numpy import ndarray

def show(img: ndarray):
    cv2.imshow('show', img)
    cv2.waitKey(0)
