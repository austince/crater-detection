from typing import Tuple
import cv2 as cv
import numpy as np
from ..util import angle_between_points, angle_between_with_origin


class Crater:
    """
    """
    def __init__(self, high_c, low_c, combinded_c):
        self.high_contour = high_c
        self.low_contour = low_c
        self.full_contour = combinded_c

    def sun_angle(self) -> np.real:
        high_pos, high_rad = cv.minEnclosingCircle(self.high_contour)
        low_pos, low_rad = cv.minEnclosingCircle(self.low_contour)

        low_pos = np.uint(np.around(low_pos))
        high_pos = np.uint(np.around(high_pos))

        btw_circles = angle_between_points(high_pos, low_pos)

        return btw_circles

    def radius(self) -> np.real:
        pos, rad = self.min_enclosing_circle()
        return rad

    def arc_length(self) -> np.real:
        return cv.arcLength(self.full_contour, True)

    def area(self):
        return cv.contourArea(self.full_contour)

    def hull(self):
        return cv.convexHull(self.full_contour)

    def bounding_rect(self):
        return cv.boundingRect(self.full_contour)

    def min_enclosing_circle(self):
        return cv.minEnclosingCircle(self.full_contour)
