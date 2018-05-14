import cv2 as cv
import numpy as np

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


def generate(num_craters=NCraters,
             width=FieldX,
             height=FieldY,
             min_radius=MinCrater,
             max_radius=MaxCrater,
             shadow_factor=CraterShadowFactor,
             alpha=Alpha,
             sun_angle=SunAngle):
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
    output_img = np.full([height, width, 3], BG_COLOR, dtype=np.uint8)

    for i in range(num_craters):
        crater_x = np.random.randint(0, width)
        crater_y = np.random.randint(0, height)

        uni = np.random.uniform(0, 1)

        crater_a = min_radius ** (Alpha + 1)
        crater_b = max_radius ** (Alpha + 1) - crater_a

        crater_real = (crater_a + (crater_b * uni)) ** (1 / (1 + alpha))
        crater_size = np.floor(crater_real)

        # draw light -> gray -> dark
        angle_rad = np.deg2rad(sun_angle)
        crater_offset_x = np.cos(angle_rad) * int(np.round(crater_size / shadow_factor))
        crater_offset_y = np.sin(angle_rad) * int(np.round(crater_size / shadow_factor))
        crater_radius = int(np.round(crater_size - (crater_size / shadow_factor / 2)))

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

    return output_img

