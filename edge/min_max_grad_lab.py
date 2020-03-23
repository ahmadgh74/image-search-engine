import cv2
import numpy as np

"""
    sobel edge detection filters
"""

gx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]], dtype=np.uint8)

gy = np.array([
    [1, 2, -1],
    [0, 0, 0],
    [-1, -2, -1]], dtype=np.uint8)


def calc_grad(num, l, a, b):
    rh = cv2.filter2D(l, cv2.CV_32F, gx[0])
    gh = cv2.filter2D(a, cv2.CV_32F, gx[1])
    bh = cv2.filter2D(b, cv2.CV_32F, gx[2])

    rv = cv2.filter2D(l, cv2.CV_32F, gy[0])
    gv = cv2.filter2D(a, cv2.CV_32F, gy[1])
    bv = cv2.filter2D(b, cv2.CV_32F, gy[2])
    gxx = rh ** 2 + gh ** 2 + bh ** 2
    gyy = rv ** 2 + gv ** 2 + bv ** 2
    gxy = rh * rv + gh * gv + bh * bv

    theta = np.round(np.arctan(2 * gxy / (gxx - gyy + .00001) / 2), 4)
    m_g1 = .5 * ((gxx + gyy) + (gxx - gyy) * np.cos(theta * 2) + 2 * gxy * np.sin(2 * (theta + (np.pi / 2))))
    m_g2 = .5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2.0 * (theta + (np.pi / 2.0))) + 2.0 * gxy * np.sin(
        2.0 * (theta + (np.pi / 2.0))))
    m_g1[m_g1 < 0] = 0
    m_g2[m_g2 < 0] = 0
    g1 = np.sqrt(m_g1)
    g2 = np.sqrt(m_g2)

    ori = np.zeros(g1.shape)
    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            if g1[i, j] >= g2[i, j]:
                direction = 90 + theta[i, j] * 180 / np.pi
                ori[i, j] = np.int32(direction * num / 360)
            else:
                direction = 180.0 + (theta[i, j] + np.pi / 2.0) * 180.0 / np.pi
                ori[i, j] = np.int32(direction * num / 360)

            if ori[i, j] >= num - 1:
                ori[i, j] = num - 1
    return np.int32(ori)
