import numpy as np
import cv2 as cv
from typing import Tuple, List, Any
from scipy.spatial import distance
from scipy.signal import argrelmax

from ..util import logger, angle_between_points
from crater_detection.models import Crater, CraterField

OUTLINE_COLOR = (0, 255, 0)
OUTLINE_THICKNESS = 3

erode_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dilate_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

# Exports
__all__ = ["detect"]


def get_peak_values(img, low_percentile=0.001, high_percentile=0.95):
    """

    :param img:
    :param low_percentile: [0.001]
    :param high_percentile: [0.95]
    :return:
    """
    flattened = img.flatten()
    peaks = argrelmax(flattened)
    peak_vals = list(map(lambda x: flattened[x], peaks))
    sorted_peaks = sorted(peak_vals[0])
    lower_bound = int(np.floor(len(sorted_peaks) * low_percentile))
    upper_bound = int(np.floor(len(sorted_peaks) * high_percentile))
    min_val = sorted_peaks[lower_bound]
    max_val = sorted_peaks[upper_bound]
    return int(min_val), int(max_val)


def clean_image(img: np.ndarray) -> np.ndarray:
    # http://opencv-python-tutroals.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html?highlight=structuring%20element
    # Clean out points by "open"-ing
    cv.erode(img, erode_kernel)
    cv.dilate(img, dilate_kernel)
    return img


def close_image(img: np.ndarray) -> np.ndarray:
    # http://opencv-python-tutroals.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html?highlight=structuring%20element
    # Clean out points by "closing"-ing
    return cv.morphologyEx(img, cv.MORPH_CLOSE, dilate_kernel)


def get_contours(img: np.ndarray) -> Tuple[List, Any]:
    contour_image, contours, hierarchy = cv.findContours(img,
                                                         # Get a tree of hierarchies to calculate crater "children"
                                                         cv.RETR_TREE,
                                                         # Though more memory intensive,
                                                         # no approx. is better for results
                                                         cv.CHAIN_APPROX_NONE,
                                                         )

    for i in range(len(contours)):
        contours[i] = np.squeeze(contours[i], axis=1)
    return contours, hierarchy


def detect(input_image: np.ndarray) -> Tuple[np.ndarray, CraterField]:
    """"
    Tests:
    - Threshold Pyramid (?), get light and dark points
    - Gaussian Pyramid, apply contour detection and Hough Circle detection on each

    Still to do:
    - Build likely-hood based on combined results
    - Build Hierarchy with combined results
    """
    # Make sure it's black and white
    if len(input_image.shape) == 2:
        # Already in grayscale
        bw_img = input_image
    else:
        bw_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # logger.info("Finding circles")
    # circles = [] # find_circles(bw_img)
    # logger.info("Found %i total circles" % len(circles))

    min_val, max_val = get_peak_values(bw_img)
    logger.debug("Lowest img value:", np.min(bw_img))
    logger.debug("Highest img value:", np.max(bw_img))

    low_thresh_image = cv.inRange(bw_img,
                                  0,
                                  min_val,
                                  )
    low_clean = close_image(low_thresh_image)

    # Get bright regions
    # high_thresh, high_thresh_image = cv.threshold(img, 254, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Invert the image so that the light parts become same as previously
    # extracted dark parts
    high_thresh_image = cv.inRange(bw_img,
                                   max_val,
                                   255,
                                   )
    high_clean = close_image(high_thresh_image)

    # Find contours in each
    low_contours, low_heirarchy = get_contours(low_clean)
    high_contours, high_heirarchy = get_contours(high_clean)

    # Merge them
    # thresh_image = cv.max(high_thresh_image, low_thresh_image)
    # closed = close_image(thresh_image)
    # clean_image(thresh_image)

    # Pair high and low contours
    # for each high contour, find the closest low contour by center
    def mapper(c):
        pos, rad = cv.minEnclosingCircle(c)
        return c, np.uint(np.around(pos))

    high_with_pos = list(map(mapper, high_contours))
    low_with_pos = list(map(mapper, low_contours))

    def get_contour_params(c):
        (x, y), rad = cv.minEnclosingCircle(c)
        return x, y, rad

    def distance_between_contours(cont, other):
        # x, y, width, height = cv.boundingRect(c)
        c_params = get_contour_params(cont)
        o_params = get_contour_params(other)
        return np.linalg.norm(np.array(c_params) - np.array(o_params))

    # Find closest low
    # Compute the distances from each high to each low
    # Pair to the closest right now, we'll see about
    # high x low => dist: real
    # More efficient alg:
    #   https://stackoverflow.com/questions/5077318/given-two-large-sets-of-points-how-can-i-efficiently-find-pairs-that-are-near
    dists = np.zeros(shape=(len(high_contours), len(low_contours)))
    h_matches = np.ndarray(shape=(len(high_contours),), dtype=np.uint)
    l_matches = np.ndarray(shape=(len(low_contours),), dtype=np.uint)
    craters = []
    logger.debug("Matching high and low crater pairs")
    for h_i, h in enumerate(high_contours):
        for l_i, l in enumerate(low_contours):
            dists[h_i][l_i] = distance_between_contours(h, l)

        min_ind = dists[h_i].argmin()
        # just assign them both for now
        h_matches[h_i] = int(min_ind)
        l_matches[min_ind] = h_i
        full_contour = np.append(high_contours[h_i], low_contours[min_ind], axis=0)
        craters.append(Crater(high_contours[h_i], low_contours[min_ind], full_contour))


    # Draw all detected contours on the image
    logger.info("Drawing craters")
    color_image = cv.cvtColor(bw_img, cv.COLOR_GRAY2BGR)

    logger.info("Drawing contours")
    cv.drawContours(color_image, low_contours, -1, (0, 0, 255), 2)
    cv.drawContours(color_image, high_contours, -1, (255, 0, 0), 2)
    # cv.drawContours(color_image, combinded, -1, (0, 255, 0), 2)
    cv.drawContours(color_image, list(map(lambda c: c.full_contour, craters)), -1, (0, 255, 0), 2)

    # logger.info("Drawing circles")
    # for circle in circles:
    #     cv.circle(color_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    logger.info("Drawing contour connections")
    for h_i, l_i in enumerate(h_matches):
        h, h_pos = high_with_pos[h_i]
        l, l_pos = low_with_pos[int(l_i)]
        cv.line(color_image, tuple(h_pos), tuple(l_pos), (0, 0, 0), 2)

    # Let's do some stats
    height, width, _ = color_image.shape
    crater_field = CraterField(width, height, craters)

    return color_image, crater_field
