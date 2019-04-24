from time import time


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.seconds = None
        self.milliseconds = None

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()
        s = self.end_time - self.start_time
        ms = (self.end_time - self.start_time) * 1000
        self.seconds = "{:.2f}s".format(s)
        self.milliseconds = "{:.2f}ms".format(ms)
