from termcolor import cprint
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Logger:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, e) -> None:
        self.enabled = e

    def set_level(self, level: str) -> None:
        self.level = level

    @staticmethod
    def _print(*args, color='white', level=None):
        if level is not None:
            args = list(args)
            args.insert(0, str(level).upper() + ":")
        cprint(" ".join(map(str, args)), color=color)

    def error(self, *args, color='red'):
        self._print(*args, color=color, level='error')

    def log(self, *args, color='white', level='log'):
        if self.enabled:
            self._print(*args, color=color, level=level)

    def info(self, *args, color='white'):
        if self.enabled:
            self.log(*args, color=color, level='info')

    def debug(self, *args, color='white'):
        if self.enabled:
            self.log(*args, color=color, level='debug')


logger = Logger()
