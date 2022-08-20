import numpy as np
from skimage import measure
import imagehash
import cv2
from scipy.spatial import distance as dist
from src.other_func import *


def mean_squared_error(imgA, imgB):
    res = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    res /= float(imgA.shape[0] * imgA.shape[1])
    return res


def hash(imgA, imgB):
    return abs(imagehash.average_hash(imgA) - imagehash.average_hash(imgB))


def euclidean(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.euclidean(arrA, arrB)


def euclidean_squared(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.sqeuclidean(arrA, arrB)


def manhattan(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.cityblock(arrA, arrB)


def canberra(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.canberra(arrA, arrB)


def minkowski(arrA, arrB, p=1):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    if p == 0:
        return "p can't be 0"
    return dist.minkowski(arrA, arrB, p)


def minkowski_weighted(arrA, arrB, p=1, weight=1):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    if p == 0:
        return "p can't be 0"
    if weight == 0:
        return "weight can't be 0"
    return dist.minkowski(arrA, arrB, p, weight)


def cosine(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.cosine(arrA, arrB)


def hamming(arrA, arrB):
    if len(arrA) != len(arrB):
        return "Parameters must have the same size"
    return dist.hamming(arrA, arrB )


def hist_correlation(histA, histB):
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)


def hist_chi_square(histA, histB):
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CHISQR)


def hist_chi_square_alt(histA, histB):
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CHISQR_ALT)


def hist_intersection(histA, histB):
    return cv2.compareHist(histA, histB, cv2.HISTCMP_INTERSECT)


def hist_bhattacharyya(histA, histB):
    return cv2.compareHist(histA, histB, cv2.HISTCMP_BHATTACHARYYA)


def hist_kullback_leibler_divergence(histA, histB):
    # histA = cv2.calcHist(imgA, [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # histB = cv2.calcHist(imgB, [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(histA, histB, cv2.HISTCMP_KL_DIV)


def construct_hist_BGR(filename):
    img = cv2.imread(filename)
    return cv2.calcHist(img, [0, 1, 2], None, [128, 128, 128], [0, 256, 0, 256, 0, 256])


def construct_hist_BGR_normalized(filename):
    img = cv2.imread(filename)
    hist = cv2.calcHist(img, [0, 1, 2], None, [128, 128, 128], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, 0, 255, cv2.NORM_MINMAX)


def construct_hist_HSV(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.calcHist(img, [0, 1, 2], None, [90, 128, 128], [0, 180, 0, 256, 0, 256])


def construct_hist_HSV_normalized(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(img, [0, 1, 2], None, [90, 128, 128], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, 0, 255, cv2.NORM_MINMAX)


def color(imgA, imgB, pixel_color_comparison=euclidean, array_image_comparison=euclidean, accuracy=1):
    # R, G, B, yellow, cyan, magenta, black, white, gray
    switcher = {
        # 3 : R, G, B
        1: [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        # 6 : R, G, B, Y, C, M
        2: [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)],
        # 9 : R, G, B, yellow, cyan, magenta, black, white, gray
        3: [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0),
            (255, 255, 255), (128, 128, 128)],
        # 16 : red, lime, blue, yellow, cyan, magenta, black, white, gray, silver, maroon, olive, green, purple, teal, navy
        4: [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0),
            (255, 255, 255), (128, 128, 128), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
            (0, 128, 128), (0, 0, 128)]
    }

    colors = switcher.get(accuracy, None)

    if colors is None:
        return "Accuracy arguments exception"

    resA = [0] * len(colors)
    resB = [0] * len(colors)
    mini = -1
    min = 1000

    for i in range(imgA.shape[0]):
        for j in range(imgA.shape[1]):
            (r, g, b) = imgA[i, j]
            for c in range(0, len(colors)):
                (cr, cg, cb) = colors[c]
                if pixel_color_comparison((cr, cg, cb), (r, g, b)) < min:
                    mini = c
                    min = pixel_color_comparison((cr, cg, cb), (r, g, b))
            resA[mini] += 1
            min = 1000

    mini = -1
    min = 1000

    for i in range(imgB.shape[0]):
        for j in range(imgB.shape[1]):
            (r, g, b) = imgB[i, j]
            for c in range(0, len(colors)):
                (cr, cg, cb) = colors[c]
                if pixel_color_comparison((cr, cg, cb), (r, g, b)) < min:
                    mini = c
                    min = pixel_color_comparison((cr, cg, cb), (r, g, b))
            resB[mini] += 1
            min = 1000

    # resoA = get_resolution_cv2(imgA)
    # resoB = get_resolution_cv2(imgB)

    resA = normalize(resA, get_resolution_cv2(imgA))
    resB = normalize(resB, get_resolution_cv2(imgA))
    """
    for i in range(0, len(resA)):
        resA[i] = (resA[i] * 100) / resoA
        resB[i] = (resB[i] * 100) / resoB
    """

    return array_image_comparison(resA, resB)


def hist(listimg, construct_hist=construct_hist_BGR, compare_hist=hist_correlation, accuracy=1):
    base = construct_hist(listimg[0])
    if len(listimg) == 0:
        return "Images set is empty"

    res = {}

    for i in range(0, len(listimg)):
        img = listimg[i]
        if i == 0:
            base = construct_hist(img)
            res[img] = compare_hist(base, base)
        else:
            hist = construct_hist(listimg[i])
            res[img] = compare_hist(base, hist)

    return sorted(res.items(), key=lambda t: t[1])
