import numpy as np
from typing import List

from ..util import logger
from . import Crater


class CraterField:
    def __init__(self, width: int, height: int, craters: List[Crater]):
        self.width = width
        self.height = height
        self.craters = craters

    def stats(self):
        logger.info("Crater Field Stats:")
        crater_rads = np.array(list(map(lambda c: c.radius(), self.craters)))
        mean_rad = np.mean(crater_rads)
        max_rad = np.max(crater_rads)
        min_rad = np.min(crater_rads)

        crater_angles = np.array(list(map(lambda c: c.sun_angle(), self.craters)))
        mean_sun_angle = np.mean(crater_angles)
        max_sun_angle = np.max(crater_angles)
        min_sun_angle = np.min(crater_angles)

        stats = {
            "width": self.width,
            "height": self.height,
            "num_craters": len(self.craters),
            "mean_rad": mean_rad,
            "max_rad": max_rad,
            "min_rad": min_rad,
            "sun_angle": mean_sun_angle,
            "min_sun_angle": min_sun_angle,
            "max_sun_angle": max_sun_angle,
            "min_sun_angle_degrees": np.rad2deg(min_sun_angle),
            "max_sun_angle_degrees": np.rad2deg(max_sun_angle),
            "sun_angle_degrees": np.rad2deg(mean_sun_angle),
        }

        return stats
