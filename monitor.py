# monitor.py
import time

class Monitor:
    """Basic latency and usage tracking."""
    def __init__(self):
        self.metrics = {}

    def start(self, key):
        self.metrics[key] = time.time()

    def stop(self, key):
        if key in self.metrics:
            latency = time.time() - self.metrics[key]
            print(f"{key} latency: {latency:.2f}s")
