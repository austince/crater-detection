import numpy as np
import cv2 as cv
from typing import Tuple, List
from scipy.misc import imshow
from matplotlib import pyplot as plt

from ..util import logger
from .models.Crater import Crater

OUTLINE_COLOR = (0, 255, 0)
OUTLINE_THICKNESS = 3


def create_gaussian_pyramid(img: np.ndarray, steps=6) -> List[np.ndarray]:
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


def scale_contour(contour) -> np.ndarray:
    pass


def image_info(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def detect(input_image: np.ndarray) -> Tuple[np.ndarray, List[Crater]]:
    """"
    Methodology:
    - Threshold Pyramid (?), get light and dark points
    - Gaussian Pyramid, apply contour detection on each
    - Build likely-hood based on combined results
    - Build Hierarchy with combined results
    """
    src_height, src_width, _ = input_image.shape
    bw_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # Equalize contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bw_image = clahe.apply(bw_image)

    gauss_pyr = create_gaussian_pyramid(bw_image)
    craters = []

    for i, img in enumerate(gauss_pyr):
        img_height, img_width = img.shape
        # Mark a ratio so we can remap the detected craters to the full scale image
        height_ratio: float = img_height / src_height
        width_ratio: float = img_width / src_width

        logger.debug(np.min(img))
        logger.debug(np.max(img))
        # Get dark regions
        logger.log("Thresholding image %i of %i" % (i, len(gauss_pyr)))
        low_thresh_image = cv.inRange(img,
                                      0,
                                      40,
                                      )

        # Get bright regions
        # high_thresh, high_thresh_image = cv.threshold(img, 254, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        high_thresh_image = cv.inRange(img,
                                       200,
                                       255,
                                       )
        # Merge them
        thresh_image = cv.max(high_thresh_image, low_thresh_image)

        blurred_image: np.ndarray = cv.GaussianBlur(thresh_image, (9, 9), sigmaX=2, sigmaY=2)

        erode_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dilate_kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

        # http://opencv-python-tutroals.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html?highlight=structuring%20element
        # Clean out points by "open"-ing
        cv.erode(blurred_image, erode_kernel)
        cv.dilate(blurred_image, dilate_kernel)

        contour_image, contours, hierarchy = cv.findContours(blurred_image,
                                                             # Get a tree of hierarchies to calculate crater "children"
                                                             cv.RETR_TREE,
                                                             # Though more memory intensive,
                                                             # no approx. is better for results
                                                             cv.CHAIN_APPROX_NONE,
                                                             )

        # Resize contours to map to the input image
        # Contours in form [[Point]] aka [ [ [(x_0, y_0)],  [(x_1, y_1)], [(x_n, y_n)] ] ]
        # TODO: optimize using numpy array broadcasting
        # resized_contours = []
        # for contour in contours:
        #     ratio = np.array([height_ratio, width_ratio])[:, np.newaxis]
        #     resized = np.multiply(contour.copy(), ratio, casting='unsafe')  # more like 'fun-safe'
        #     resized_contours.append(resized)

        logger.log("Detecting circles")
        cirlces = cv.HoughCircles(blurred_image,
                                  cv.HOUGH_GRADIENT,
                                  # cv.HOUGH_MULTI_SCALE, # Might be good when implemented
                                  1,
                                  img_height / 8,
                                  param1=200,
                                  param2=100,
                                  minRadius=0
                                  )

        # Create crater tree
        logger.log("Creating Crater list")
        craters = list(map(lambda cont: Crater(cont), contours))

        # Draw all detected contours on the image
        if width_ratio == 1 and height_ratio == 1:
            logger.log("Drawing craters")
            cv.drawContours(input_image, contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)

    return input_image, craters
