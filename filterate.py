# 1. Import the necessary packages
import numpy as np
import cv2
import urllib

from PIL import Image

import time, os, glob
import sys
import shutil
import random

def filterate_img(img):
#     img = cv2.imread(filepath, -1)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((31,31), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 15)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    width = 512
    height = 512
    dim = (width, height)
    resized = cv2.resize(result_norm, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray

