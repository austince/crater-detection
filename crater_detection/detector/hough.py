import numpy as np
import cv2 as cv
from typing import Tuple, List, Any
from scipy.spatial import distance
from scipy.signal import argrelmax
from matplotlib import pyplot as plt

from ..util import logger
from crater_detection.models import Crater, CraterFieldStats, CraterField


def image_info(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


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
