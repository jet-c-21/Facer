"""
author: Jet C.
GitHub: https://github.com/jet-chien
Create Date: 2021/1/9
"""
# coding: utf-8
import json
import pickle


def save_as_pkl(data: object, save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fp: str):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def load_json(fp: str) -> dict:
    return json.load(open(fp, 'r', encoding='utf-8'))


def save_as_json(data, fp: str):
    json.dump(data, open(fp, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
