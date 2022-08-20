import operator
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.img import base
from src.other_func import read_image
from src.distance import *


def main():
    print("BEGIN")

    start = time.time()

    moon = cv2.imread('img/moon1000x1000.jpg')
    planet = cv2.imread('img/planete1000x1000.jpg')

    imgb = ['img/base/sun1.jpg', 'img/base/sun2.jpg', 'img/base/sun3.jpg', 'img/base/sun4.jpg', 'img/base/sun5.jpg',
            'img/base/sun6.jpg', 'img/base/sun7.jpg', 'img/base/sun8.jpg', 'img/base/sun9.jpg', 'img/base/sun10.jpg',
            'img/base/sun11.jpg', 'img/base/sun12.jpg']

    res = {}
    # res2 = {}
    # res = hist(imgb, construct_hist_BGR, hist_correlation)
    # res2 = hist(imgb, construct_hist_HSV_normalized, hist_correlation)


    #base = cv2.imread(imgb[0])
    base = read_image(imgb[0])
    # (h, w, c) = base.shape

    for i in range(0, len(imgb)):
        # img = cv2.imread(imgb[i])
        img = read_image(imgb[i])
        # img = resize_image(img, h, w)
        # res[i] = hamming(img_to_1d_array(base), img_to_1d_array(img))
        res[i] = hash(base, img)

    res = sorted(res.items(), key=lambda t: t[1])

    end = time.time()
    print("TIME = " + str(end - start))
    print("res = " + str(res))
    # print("res2 = " + str(res2))

    print("END")


main()
