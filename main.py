import itertools
import timeit

import matplotlib.pyplot as plt
from scipy.spatial.distance import canberra

from histogram import process
from utils import read_json, write_json


def test_main(path):
    return process(path)


def process_images():
    hists = read_json()
    latest = int(max(hists.keys() if hists.keys() else 0, 1))
    max_ = max(10001, latest + 1000)
    for i in range(latest, max_):
        hists[i] = list(test_main(f'data/corel10K/{i}.jpg'))
        write_json(hists)


def calc_nearest_to(nearest):
    pics = read_json()
    selected = pics.pop(str(nearest))
    nearest_pics = {}
    for k, v in pics.items():
        nearest_pics[k] = canberra(selected, v)
        nearest_pics = {a: b for a, b in sorted(nearest_pics.items(), key=lambda item: item[1])}
        nearest_pics = dict(itertools.islice(nearest_pics.items(), 20))

    return nearest_pics


def plt_compare(one, two):
    hists = read_json()
    hist1 = hists[str(one)]
    hist2 = hists[str(two)]
    plt.plot(hist1, color='red')
    similarity = canberra(hist1, hist2)
    print(f'length : {len(hist2)}')
    print(f'similarity : {similarity}')
    # plt.figure()
    plt.plot(hist2)
    plt.show()


if __name__ == '__main__':
    # edge_detection()
    start = timeit.default_timer()
    # nearest = calc_nearest_to(1)
    plt_compare(1, 5081)
    # print(nearest)
    stop = timeit.default_timer()
    print(f'time calc {stop - start}')
