from typing import Tuple
import cv2 as cv
import numpy as np


class Crater:
    """
    Think of like a tree node
    """
    def __init__(self, contour):
        self.contour = contour
        # (x, y), rad = self.min_enclosing_circle()
        # self.pos = np.array([x, y])
        self.children = []

    def contains_point(self, pt: Tuple[float, float]) -> bool:
        pass

    def arc_length(self):
        return cv.arcLength(self.contour, True)

    def area(self):
        return cv.contourArea(self.contour)

    def hull(self):
        return cv.convexHull(self.contour)

    def bounding_rect(self):
        return cv.boundingRect(self.contour)

    def min_enclosing_circle(self):
        return cv.minEnclosingCircle(self.contour)
