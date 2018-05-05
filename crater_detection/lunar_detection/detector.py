import numpy as np
import cv2 as cv
from typing import Tuple, List
from scipy.misc import imshow
from scipy.signal import argrelmax
from matplotlib import pyplot as plt

from ..util import logger
from .models.Crater import Crater

OUTLINE_COLOR = (0, 255, 0)
OUTLINE_THICKNESS = 3

# Exports
__all__ = ["detect"]


def create_gaussian_pyramid(img: np.ndarray, steps=1) -> List[np.ndarray]:
    """
    :see: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    :param img:
    :param steps:
    :return:
    """
    cur_img = img.copy()
    pyr = [cur_img]

    for i in range(steps - 1):
        cur_img = cv.pyrDown(cur_img)
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


def get_contours(img):
    contour_image, contours, hierarchy = cv.findContours(img,
                                                         # Get a tree of hierarchies to calculate crater "children"
                                                         cv.RETR_TREE,
                                                         # Though more memory intensive,
                                                         # no approx. is better for results
                                                         cv.CHAIN_APPROX_NONE,
                                                         )
    return contours, hierarchy


def detect(input_image: np.ndarray) -> Tuple[np.ndarray, List[Crater]]:
    """"
    Methodology:
    - Threshold Pyramid (?), get light and dark points
    - Gaussian Pyramid, apply contour detection on each
    - Build likely-hood based on combined results
    - Build Hierarchy with combined results
    """
    if len(input_image.shape) == 2:
        # Already in grayscale
        src_height, src_width = input_image.shape
        bw_image = input_image
    else:  # must be 3
        src_height, src_width, _ = input_image.shape
        bw_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # Equalize contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    logger.log(get_peak_values(bw_image))
    # bw_image = clahe.apply(bw_image)

    gauss_pyr = create_gaussian_pyramid(bw_image)
    craters = []

    for i, img in enumerate(gauss_pyr):
        img_height, img_width = img.shape
        # Mark a ratio so we can remap the detected craters to the full scale image
        height_ratio: float = img_height / src_height
        width_ratio: float = img_width / src_width
        # logger.log("Normalizing contrast")
        # img = clahe.apply(img)

        min_val, max_val = get_peak_values(img)
        logger.debug(np.min(img))
        logger.debug(np.max(img))

        logger.log("Thresholding image %i of %i" % (i, len(gauss_pyr)))
        # Todo: Find thresholds dynamically based on lowest and highest peaks
        low_thresh_image = cv.inRange(img,
                                      0,
                                      min_val,
                                      )
        low_clean = close_image(low_thresh_image)

        # Get bright regions
        # high_thresh, high_thresh_image = cv.threshold(img, 254, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Invert the image so that the light parts become same as previously
        # extracted dark parts
        high_thresh_image = cv.inRange(img,
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

        blurred_image: np.ndarray = cv.GaussianBlur(thresh_image, (9, 9), sigmaX=2, sigmaY=2)



        # Resize contours to map to the input image
        # Contours in form [[Point]] aka [ [ [(x_0, y_0)],  [(x_1, y_1)], [(x_n, y_n)] ] ]
        # TODO: optimize using numpy array broadcasting
        # resized_contours = []
        # for contour in contours:
        #     ratio = np.array([height_ratio, width_ratio])[:, np.newaxis]
        #     resized = np.multiply(contour.copy(), ratio, casting='unsafe')  # more like 'fun-safe'
        #     resized_contours.append(resized)

        # TODO: Hough Circles?
        # logger.log("Detecting circles")
        # cirlces = cv.HoughCircles(blurred_image,
        #                           cv.HOUGH_GRADIENT,
        #                           # cv.HOUGH_MULTI_SCALE, # Might be good when implemented
        #                           1,
        #                           img_height / 8,
        #                           param1=200,
        #                           param2=100,
        #                           minRadius=0
        #                           )

        # Create crater tree
        logger.log("Creating Crater list")
        # craters = list(map(lambda cont: Crater(cont), contours))

        # Draw all detected contours on the image
        if width_ratio == 1 and height_ratio == 1:
            logger.log("Drawing craters")
            cv.drawContours(input_image, low_contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)
            cv.drawContours(input_image, high_contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)

    return input_image, craters
