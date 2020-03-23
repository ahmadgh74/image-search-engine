import numpy as np
from numba import jit


@jit()
def calc_value(d1, d2, d3):
    return np.sqrt(d1 ** 2 + d2 ** 2 + d3 ** 2)


@jit()
def compute(colors, orientation, lab, csa, csb, d):
    matrix = np.zeros((csa + csb))

    # ----------------- direction=0 ------------------------

    for i in range(colors.shape[0]):
        for j in range(colors.shape[1] - d):
            if orientation[i, j + d] == orientation[i, j]:
                value = calc_value(lab[i, j + d, 0] - lab[i, j, 0],
                                   lab[i, j + d, 1] - lab[i, j, 1],
                                   lab[i, j + d, 2] - lab[i, j, 2])
                matrix[colors[i, j]] += value

            if colors[i, j + d] == colors[i, j]:
                value = calc_value(lab[i, j + d, 0] - lab[i, j, 0],
                                   lab[i, j + d, 1] - lab[i, j, 1],
                                   lab[i, j + d, 2] - lab[i, j, 2])
                matrix[orientation[i, j] + csa] += value

    # ----------------- direction=90 ------------------------

    for i in range(colors.shape[0] - d):
        for j in range(colors.shape[1]):
            if orientation[i + d, j] == orientation[i, j]:
                value = calc_value(lab[i + d, j, 0] - lab[i, j, 0],
                                   lab[i + d, j, 1] - lab[i, j, 1],
                                   lab[i + d, j, 2] - lab[i, j, 2])
                matrix[colors[i, j]] += value

            if colors[i + d, j] == colors[i, j]:
                value = calc_value(lab[i + d, j, 0] - lab[i, j, 0],
                                   lab[i + d, j, 1] - lab[i, j, 1],
                                   lab[i + d, j, 2] - lab[i, j, 2])
                matrix[orientation[i, j] + csa] += value

    # ----------------- direction=135 ------------------------

    for i in range(colors.shape[0] - d):
        for j in range(colors.shape[1] - d):
            if orientation[i + d, j + d] == orientation[i, j]:
                value = calc_value(lab[i + d, j + d, 0] - lab[i, j, 0],
                                   lab[i + d, j + d, 1] - lab[i, j, 1],
                                   lab[i + d, j + d, 2] - lab[i, j, 2])
                matrix[colors[i, j]] += value

            if colors[i + d, j + d] == colors[i, j]:
                value = calc_value(lab[i + d, j + d, 0] - lab[i, j, 0],
                                   lab[i + d, j + d, 1] - lab[i, j, 1],
                                   lab[i + d, j + d, 2] - lab[i, j, 2])
                matrix[orientation[i, j] + csa] += value

    # ----------------- direction=135 ------------------------

    for i in range(colors.shape[0]):
        for j in range(colors.shape[1] - d):
            if orientation[i - d, j + d] == orientation[i, j]:
                value = calc_value(lab[i - d, j + d, 0] - lab[i, j, 0],
                                   lab[i - d, j + d, 1] - lab[i, j, 1],
                                   lab[i - d, j + d, 2] - lab[i, j, 2])
                matrix[colors[i, j]] += value

            if colors[i - d, j + d] == colors[i, j]:
                value = calc_value(lab[i - d, j + d, 0] - lab[i, j, 0],
                                   lab[i - d, j + d, 1] - lab[i, j, 1],
                                   lab[i - d, j + d, 2] - lab[i, j, 2])
                matrix[orientation[i, j] + csa] += value

    hist = matrix / 4
    return hist
