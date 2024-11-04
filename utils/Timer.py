import logging
import time


class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        logging.info(f"Время выполнения: {self.interval:.6f} секунд")

