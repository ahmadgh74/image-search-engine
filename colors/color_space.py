from functools import partial

import cv2
import numpy as np


def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    l, a, b = np.float32(l / 255 * 100), np.float32(a - 128), np.float32(b - 128)
    return cv2.merge([l, a, b])


def func(value, col):
    if value >= col - 1:
        return col - 1
    if value < 0:
        return 0
    return value


def color_quantization(lab, col1, col2, col3):
    l, a, b = cv2.split(lab)
    new_l = np.int8(l * col1 / 100)
    new_a = np.int8((a + 127) * col2 / 254)
    new_b = np.int8((b + 127) * col3 / 254)
    l_func = partial(func, col=col1)
    a_func = partial(func, col=col2)
    b_func = partial(func, col=col3)
    l_vfunc = np.vectorize(l_func)
    a_vfunc = np.vectorize(a_func)
    b_vfunc = np.vectorize(b_func)
    final_l = l_vfunc(new_l)
    final_a = a_vfunc(new_a)
    final_b = b_vfunc(new_b)

    img = (col3 * col2) * final_l + col3 * final_a + final_b
    return img
