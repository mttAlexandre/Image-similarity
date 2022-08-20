import numpy as np
import PIL
from PIL import Image, ImageFilter
import cv2
from matplotlib import pyplot as plt


def read_image(path):
    try:
        img = PIL.Image.open(path)
        return img
    except Exception as e:
        print(e)


def get_resolution(image):
    return image.size


def get_resolution_cv2(image):
    return image.shape[0]*image.shape[1]


def resize_image(image, height, width):
    return cv2.resize(image, (width, height))


def resize_two_images(imgA, imgB):
    ha = imgA.shape[0]
    wa = imgA.shape[1]
    hb = imgB.shape[0]
    wb = imgB.shape[1]
    if wa == wb and ha == hb:
        return imgA, imgB
    if wa > wb:
        if ha > hb:
            resize_image(imgA, hb, wb)
        else:
            resize_image(imgA, ha, wb)
            resize_image(imgB, ha, wb)
    else:
        if ha > hb:
            resize_image(imgA, hb, wa)
            resize_image(imgB, hb, wa)
        else:
            resize_image(imgB, ha, wa)
    return imgA, imgB


def crop(image, left, top, right, bottom):
    return image.crop((left, top, right, bottom))


def center_image(image):
    width, height = image.size
    left = width / 4
    top = height / 4
    right = 3 * width / 4
    bottom = 3 * height / 4
    return left, top, right, bottom


def rotate_image(image, angle):
    return image.rotate(angle)


def grayscale(image):
    # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return image.convert("L")


def binary(image, threshold):
    gs = grayscale(image)
    arr = np.array(gs)

    for i in range(0, len(arr)):
        for j in range(0, len(arr[i])):
            if arr[i][j] >= threshold:
                arr[i][j] = 255
            else:
                arr[i][j] = 0
    return PIL.Image.fromarray(arr)


def inverse(image):
    arr = np.array(image)

    for i in range(0, len(arr)):
        for j in range(0, len(arr[i])):
            # pixel = img.getpixel((j, i))
            pixel = arr[i][j]
            p = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
            arr[i][j] = p
    return PIL.Image.fromarray(arr)


def symetry(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def contour(image):
    return image.filter(ImageFilter.CONTOUR)


def img_to_1d_array(img):
    return np.array(img).flatten()


def main():
    print("BEGIN TEST")

    # IMAGE DE BASE
    # img = read_image('img/img2.jpg')
    # img.show()
    # print(img.size)
    # img2 = resize_image(img, 200, 200)
    # img2.show()

    # img_array = np.array(img)
    # print(img_array)
    # print(img_array.shape)

    # center = center_image(img)
    # print(center)
    # left, top, right, bottom = center
    # center_crop = crop(img, left, top, right, bottom)
    # center_crop.show()

    # IMAGE EN NIVEAU DE GRIS
    # gs = grayscale(img)
    # gs.show()

    # IMAGE EN BINAIRE
    # bin = binary(img, 100)
    # bin.show()

    # IMAGE EN BINAIRE
    # binv2 = binary(img, 200)
    # binv2.show()

    # INVERSE DE L'IMAGE
    # inv = inverse(img)
    # inv.show()

    # IMAGE EN SIMETRIQUE
    # sym = symetry(img)
    # sym.show()

    # DETECTION DE CONTOUR
    # cont = contour(img)
    # cont.show()

    # img.close()

    print("END TEST")

