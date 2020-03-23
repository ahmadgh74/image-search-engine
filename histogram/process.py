import cv2

from colors import rgb_to_lab, color_quantization
from edge import calc_grad
from histogram import compute


def process(path):
    image = cv2.imread(path)
    return process_image(image)


def process_image(image):
    lnum = 10
    anum = 3
    bnum = 3

    cnum = lnum * anum * bnum
    onum = 18
    d = 1

    lab = rgb_to_lab(image)
    l, a, b = cv2.split(lab)
    color_quantize = color_quantization(lab, lnum, anum, bnum)
    orientation = calc_grad(onum, l, a, b)
    hist = compute(color_quantize, orientation, lab, cnum, onum, d)
    return hist
