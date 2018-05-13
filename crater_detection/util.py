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
