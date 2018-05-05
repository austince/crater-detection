from termcolor import cprint


class Logger:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, e):
        self.enabled = e

    @staticmethod
    def _print(*args, color='white', level=None):
        if level is not None:
            args = list(args)
            args.insert(0, str(level).upper() + ":")
        cprint(" ".join(map(str, args)), color=color)

    def log(self, *args, color='white', level='info'):
        if self.enabled:
            self._print(*args, color=color, level=level)

    def debug(self, *args, color='white'):
        self._print(*args, color=color, level='debug')


logger = Logger()
