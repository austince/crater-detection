from termcolor import cprint


class Logger:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, e):
        self.enabled = e

    def _print(self, *args, color='white'):
        cprint(" ".join(map(str, *args)), color=color)

    def log(self, *args, color='white'):
        if self.enabled:
            self._print(*args, color=color)

    def debug(self, *args, color='white'):
        self._print(*args, color=color)


logger = Logger()
