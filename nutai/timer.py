import time

class Timer:
    def __init__(self, message=None):
        self.message = message
    def __enter__(self):
        self.start = time.process_time()
        return self
    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start
        if self.message:
            print(self.message, ":", self.interval)
