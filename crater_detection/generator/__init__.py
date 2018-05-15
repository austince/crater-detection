import cv2 as cv
import numpy as np
from typing import Tuple, Dict

__all__ = ["generate"]

# Defaults
SunAngle = 0

FieldX = 1024
FieldY = 1024
NCraters = 700

Alpha = -1.5
MinCrater = 10
MaxCrater = 30
CraterShadowFactor = 5

SHADOW_COLOR = (0, 0, 0)
LIGHT_COLOR = (255, 255, 255)
BG_COLOR = (100, 100, 100)


def generate(num_craters: int = NCraters,
             width: int=FieldX,
             height: int=FieldY,
             min_radius: float=MinCrater,
             max_radius: float=MaxCrater,
             shadow_factor: float=CraterShadowFactor,
             alpha: float=Alpha,
             rand_seed = None,
             sun_angle: float=SunAngle) -> Tuple[np.ndarray, Dict]:
    """
    :param num_craters:
    :param width: in px
    :param height: in px
    :param min_radius: in px
    :param max_radius: in px
    :param shadow_factor:
    :param alpha:
    :param sun_angle: in degrees
    :return:
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    output_img = np.full([height, width, 3], BG_COLOR, dtype=np.uint8)
    output_radii = []

    angle_rad = np.deg2rad(sun_angle)
    crater_a = min_radius ** (Alpha + 1)
    crater_b = max_radius ** (Alpha + 1) - crater_a

    for i in range(num_craters):
        crater_x = np.random.randint(0, width)
        crater_y = np.random.randint(0, height)

        uni = np.random.uniform(0, 1)

        crater_real = (crater_a + (crater_b * uni)) ** (1 / (1 + alpha))
        crater_size = np.floor(crater_real)

        # draw light -> gray -> dark
        crater_offset_x = np.cos(angle_rad) * int(np.round(crater_size / shadow_factor))
        crater_offset_y = np.sin(angle_rad) * int(np.round(crater_size / shadow_factor))
        crater_radius = int(np.round(crater_size - (crater_size / shadow_factor / 2)))

        output_radii.append(crater_radius)

        # Light
        cv.circle(output_img,
                  (int(crater_x - crater_offset_x), int(crater_y - crater_offset_y)),
                  crater_radius,
                  LIGHT_COLOR,
                  cv.FILLED,
                  cv.LINE_AA,  # line type
                  )

        # Shadow
        cv.circle(output_img,
                  (int(crater_x + crater_offset_x), int(crater_y + crater_offset_y)),
                  crater_radius,
                  SHADOW_COLOR,
                  cv.FILLED,
                  cv.LINE_AA,  # line type
                  )

        # Background in the middle
        cv.circle(output_img,
                  (crater_x, crater_y),
                  crater_radius,
                  BG_COLOR,
                  cv.FILLED,
                  cv.LINE_AA,  # line type
                  )

    raddi_arr = np.array(output_radii)
    stats = {
        "min_rad": np.min(raddi_arr),
        "max_rad": np.max(raddi_arr),
        "mean_rad": np.mean(raddi_arr),
        "width": width,
        "height": height,
        "num_craters": num_craters,
        "shadow_factor": shadow_factor,
        "sun_angle_degrees": sun_angle,
        "alpha": alpha,
    }

    return output_img, stats
