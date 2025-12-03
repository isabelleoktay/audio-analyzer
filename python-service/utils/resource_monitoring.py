import os 
import logging
import time
import psutil
import threading

class ResourceMonitor:
    def __init__(self, interval=0.1, baseline_window_sec=0.3):
        self.interval = interval
        self.running = False
        self.baseline_window_sec = baseline_window_sec

        self.cpu_samples = []
        self.cpu_per_core_samples = []
        self.cpu_freq_samples = []
        self.cpu_load_samples = []
        self.cpu_times_samples = []
        self.thread_count_samples = []
        self.ram_samples = []

        # Baseline values
        self.cpu_baseline = None
        self.ram_baseline = None
        self.cpu_per_core_baseline = None

    def sample_baseline(self):
        """
        Take a short-window baseline measurement of CPU (total & per-core) and RAM.
        """
        cpu_baseline_samples = []
        ram_baseline_samples = []
        per_core_samples = []

        start = time.time()
        while time.time() - start < self.baseline_window_sec:
            cpu_baseline_samples.append(psutil.cpu_percent(interval=None))
            ram_baseline_samples.append(psutil.virtual_memory().percent)
            per_core_samples.append(psutil.cpu_percent(interval=None, percpu=True))

        self.cpu_baseline = sum(cpu_baseline_samples) / len(cpu_baseline_samples)
        self.ram_baseline = sum(ram_baseline_samples) / len(ram_baseline_samples)

        # Average per-core baseline
        per_core_arr = list(zip(*per_core_samples))  # list of per-core lists
        self.cpu_per_core_baseline = [sum(core_vals)/len(core_vals) for core_vals in per_core_arr]

    def _sample(self):
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.cpu_per_core_samples.append(psutil.cpu_percent(interval=None, percpu=True))

            freq = psutil.cpu_freq()
            if freq:
                self.cpu_freq_samples.append({"current": freq.current, "min": freq.min, "max": freq.max})

            load1, load5, load15 = psutil.getloadavg()
            self.cpu_load_samples.append({"1m": load1, "5m": load5, "15m": load15})

            self.cpu_times_samples.append(psutil.cpu_times()._asdict())
            self.thread_count_samples.append(psutil.cpu_count(logical=True))
            self.ram_samples.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def start(self):
        self.sample_baseline()  # short-window baseline
        self.running = True
        self.thread = threading.Thread(target=self._sample)
        self.thread.start()
        self.start_time = time.time()

    def stop(self):
        self.running = False
        self.thread.join()
        self.end_time = time.time()

    def summary(self, feature_type):
        def avg(arr):
            return sum(arr) / len(arr) if arr else None

        cpu_avg = avg(self.cpu_samples)
        ram_avg = avg(self.ram_samples)

        # Per-core average during sampling
        per_core_avg = ([avg(core) for core in zip(*self.cpu_per_core_samples)]
                        if self.cpu_per_core_samples else None)

        # Per-core above baseline
        per_core_above_baseline = ( [c - b for c, b in zip(per_core_avg, self.cpu_per_core_baseline)]
                                    if per_core_avg and self.cpu_per_core_baseline else None )

        return {
            "feature_type": feature_type,
            "time_seconds": self.end_time - self.start_time,
            "cpu_before_start_percent": self.cpu_baseline,
            "cpu_overall_avg": cpu_avg,
            "cpu_peak": max(self.cpu_samples) if self.cpu_samples else None,
            "cpu_overall_avg_above_baseline": cpu_avg - self.cpu_baseline if cpu_avg is not None else None,
            "ram_before_start_percent": self.ram_baseline,
            "ram_avg": ram_avg,
            "ram_peak": max(self.ram_samples) if self.ram_samples else None,
            "ram_avg_above_baseline": ram_avg - self.ram_baseline if ram_avg is not None else None,
            "cpu_per_core_last": self.cpu_per_core_samples[-1] if self.cpu_per_core_samples else None,
            "cpu_per_core_avg": per_core_avg,
            "cpu_per_core_avg_above_baseline": per_core_above_baseline
        }


def get_resource_logger(log_filename="resource_use_metrics.log"):
    """
    Creates or returns a shared logger for computational resource logging.
    Ensures no duplicate handlers and guarantees log directory exists.
    """

    # Build log path: project_root/logs/<filename>
    log_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..", "logs", log_filename
    ))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("computational_resource_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevents double-printing

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger