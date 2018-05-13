import numpy as np
import cv2 as cv
from typing import Tuple, List, Any
from scipy.misc import imshow
from scipy.spatial import distance
from scipy.signal import argrelmax
from matplotlib import pyplot as plt
import itertools
import functools

from ..util import logger
from .models.Crater import Crater

OUTLINE_COLOR = (0, 255, 0)
OUTLINE_THICKNESS = 3

# Exports
__all__ = ["detect"]


def create_gaussian_pyramid(img: np.ndarray, steps=4) -> List[np.ndarray]:
    """
    :see: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    :param img:
    :param steps:
    :return:
    """
    cur_img = img.copy()
    pyr = []

    for i in range(int(steps / 2)):
        cur_img = cv.pyrDown(cur_img)
        pyr.append(cur_img)

    pyr.reverse()
    cur_img = img.copy()
    pyr.append(img.copy())

    for i in range(int(steps / 2)):
        cur_img = cv.pyrUp(cur_img)
        pyr.append(cur_img)

    return pyr


def scale_contour(contour: np.ndarray) -> np.ndarray:
    return contour


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


def image_info(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


erode_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dilate_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))


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


def dedup_circles(circles: list, min_dist: float):
    # Do not care about speed, just correctness
    final_circles = []

    get_circle_center = lambda c: (c[0], c[1])
    logger.debug("De-duplicating found craters with min_dist %f" % min_dist)
    logger.debug("Before de-deuplication: %i" % len(circles))

    def is_possible_dup(circle: Tuple[int, int, int], other: Tuple[int, int, int]):
        """
        Make the min distance lower depending on the size of the radius
        :param circle:
        :param other:
        :return:
        """
        rad = circle[2]
        other_rad = other[2]
        scaled_min_dist = min_dist * (1.01 ** np.mean([rad, other_rad]))

        # return distance.euclidean(get_circle_center(circle), get_circle_center(other)) < scaled_min_dist
        return distance.euclidean(circle, other) < scaled_min_dist

    for current_ind, circle in enumerate(circles):
        dups = [(ind, other_circle) for ind, other_circle
                in enumerate(circles[current_ind:])
                # if current_ind != ind and distance.euclidean(circle, other_circle) < MIN_DIST
                if current_ind != ind and is_possible_dup(circle, other_circle)
                ]

        # average them all
        if len(dups) > 0:
            circle = np.uint16(np.around(
                np.mean(
                    [circle] + [c for ind, c in dups],
                    axis=0
                )))

        final_circles.append(circle)

        # Remove them from list
        for i, (dup_i, dup) in enumerate(dups):
            # first is dup_i - 0
            # next is dup_i - 1, as one got deleted
            # n is dup_i - (n-1), as one got deleted
            del circles[dup_i - i]

    logger.debug("After de-deuplication:", len(circles))
    return final_circles


def find_circles(img: np.ndarray):
    blurred_image: np.ndarray = cv.GaussianBlur(img, (9, 9), sigmaX=2, sigmaY=2)
    src_height, src_width = img.shape
    gauss_pyr = create_gaussian_pyramid(blurred_image, steps=3)
    min_dup_dist = (src_height + src_width) / 2 / 500

    all_circles = []

    for i, scaled_img in enumerate(gauss_pyr):
        scale_height, scale_width = scaled_img.shape
        logger.info(
            "Detecting circles in image %i of %i at scale %i x %i" % (i + 1, len(gauss_pyr), scale_width, scale_height))
        # Mark a ratio so we can remap the detected craters to the full scale image
        height_ratio: float = scale_height / src_height
        width_ratio: float = scale_width / src_width
        wh_ratio = height_ratio / width_ratio
        wh_avg = (src_height + src_width) / 2

        circles = cv.HoughCircles(scaled_img,
                                  cv.HOUGH_GRADIENT,
                                  # cv.HOUGH_MULTI_SCALE, # Might be good when implemented
                                  1,  # dp
                                  # 20,
                                  5,  # min distance
                                  # param1=200,
                                  # param2=100,
                                  param1=20,  # passed to Canny
                                  param2=70,  # Accumulator thresh
                                  minRadius=0,
                                  maxRadius=int(wh_avg / 4),
                                  )

        if circles is not None:
            circles = circles[0]
            logger.debug("Num circles", len(circles))
            # now in format [ [y, x, radius] ... ]
            # scale them to match the original input dimens
            scaled_circles = circles / np.array([
                height_ratio,
                width_ratio,
                wh_ratio
            ])
            # round em out
            scaled_circles = np.uint16(np.around(scaled_circles))
            all_circles.extend(scaled_circles)

        else:
            logger.debug("No circles found")

    return dedup_circles(all_circles, min_dup_dist)


def closest_circle(contour_pos, circles):
    current_min = np.Infinity
    current_nearest = None
    nearest_i = None

    for i, c in enumerate(circles):
        dist = distance.euclidean(contour_pos, (c[0], c[1]))
        if dist < current_min:
            current_min = dist
            current_nearest = c
            nearest_i = i

    if current_nearest is not None:
        logger.debug("Found closest circle", "at dist", current_min)
    return current_nearest


def detect(input_image: np.ndarray) -> Tuple[np.ndarray, List[Crater]]:
    """"
    Methodology:
    - Threshold Pyramid (?), get light and dark points
    - Gaussian Pyramid, apply contour detection on each
    - Build likely-hood based on combined results
    - Build Hierarchy with combined results
    """
    # Make sure it's black and white
    if len(input_image.shape) == 2:
        # Already in grayscale
        bw_img = input_image
    else:
        bw_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    craters = []

    logger.info("Finding circles")
    circles = find_circles(bw_img)
    logger.info("Found %i total circles" % len(circles))

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
    thresh_image = cv.max(high_thresh_image, low_thresh_image)
    closed = close_image(thresh_image)
    clean_image(thresh_image)

    # Create crater tree
    logger.info("Creating Crater list")
    # craters = list(map(lambda cont: Crater(cont), contours))

    # Draw all detected contours on the image
    logger.info("Drawing craters")
    color_image = cv.cvtColor(bw_img, cv.COLOR_GRAY2BGR)

    # Pair high and low contours
    # for each high contour, find the closest low contour by center
    def mapper(c):
        pos, rad = cv.minEnclosingCircle(c)
        x, y, width, height = cv.boundingRect(c)
        avg_pos = np.mean(c, axis=0)

        return c, np.uint(np.around(pos))

    def sort_by_y(l):
        return sorted(l, key=lambda pt: pt[1][1])

    high_with_pos = list(map(mapper, high_contours))
    low_with_pos = list(map(mapper, low_contours))

    high_with_pos = sort_by_y(high_with_pos)
    low_with_pos = sort_by_y(low_with_pos)

    distances = []


    high_low_pairs = []
    combinded = []

    closest_circles = []
    for (c, c_pos) in high_with_pos + low_with_pos:
        closest_circles.append((c_pos, closest_circle(c_pos, circles)))

    for h, pos in high_with_pos:
        current_min = np.Infinity
        current_nearest = None
        nearest_i = None

        for i, (l, l_pos) in enumerate(low_with_pos):
            dist = distance.euclidean(pos, l_pos)
            if dist < current_min:
                current_min = dist
                current_nearest = (l, l_pos)
                nearest_i = i

        if current_nearest is not None:
            logger.debug("Found closest at dist", current_min)
            combinded.append(np.append(h, current_nearest[0], axis=0))
            high_low_pairs.append(((h, pos), current_nearest))
            # del low_with_pos[nearest_i]


    logger.info("Drawing contours")
    cv.drawContours(color_image, low_contours, -1, (0, 0, 255), 2)
    cv.drawContours(color_image, high_contours, -1, (255, 0, 0), 2)
    # cv.drawContours(color_image, combinded, -1, (0, 255, 0), 2)

    logger.info("Drawing circles")
    for circle in circles:
        cv.circle(color_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    # logger.info("Drawing contour connections")
    # for (h, h_pos), (l, l_pos) in high_low_pairs:
    #     cv.line(color_image, tuple(h_pos), tuple(l_pos), (0, 0, 0), 2)
    # for c_pos, circ in closest_circles:
        # cv.line(color_image, tuple(c_pos), (circ[0], circ[1]), (0, 0, 0), 2)

    return color_image, craters
