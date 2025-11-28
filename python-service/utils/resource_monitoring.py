import os 
import logging
import time
import psutil
import threading

# # Try to import NVML safely for GPU monitoring
# try:
#     import pynvml
#     pynvml.nvmlInit()
#     NVML_AVAILABLE = True
# except Exception:
#     NVML_AVAILABLE = False
#     pynvml = None


class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False

        self.cpu_samples = []
        self.cpu_per_core_samples = []
        self.cpu_freq_samples = []
        self.cpu_load_samples = []
        self.cpu_times_samples = []
        self.thread_count_samples = []

        self.ram_samples = []
        # self.gpu_util_samples = []
        # self.gpu_mem_samples = []

        # # GPU detection
        # self.gpu_handle = None
        # if NVML_AVAILABLE:
        #     try:
        #         self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        #     except Exception:
        #         self.gpu_handle = None

    def _sample(self):

        while self.running:
            # --- CPU METRICS ---

            # Total CPU %
            self.cpu_samples.append(psutil.cpu_percent(interval=None))

            # Per-core %
            self.cpu_per_core_samples.append(psutil.cpu_percent(interval=None, percpu=True))

            # CPU frequency
            freq = psutil.cpu_freq()
            if freq:
                self.cpu_freq_samples.append({
                    "current": freq.current,
                    "min": freq.min,
                    "max": freq.max,
                })

            # Load averages
            load1, load5, load15 = psutil.getloadavg()
            self.cpu_load_samples.append({
                "1m": load1,
                "5m": load5,
                "15m": load15
            })

            # CPU times breakdown
            cpu_times = psutil.cpu_times()
            self.cpu_times_samples.append(cpu_times._asdict())

            # Thread count
            self.thread_count_samples.append(psutil.cpu_count(logical=True))

            # --- RAM ---
            self.ram_samples.append(psutil.virtual_memory().percent)

            # # --- GPU (optional) ---
            # if self.gpu_handle:
            #     try:
            #         util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            #         mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

            #         self.gpu_util_samples.append(util.gpu)
            #         self.gpu_mem_samples.append(mem.used / (1024 * 1024))
            #     except Exception:
            #         pass

            time.sleep(self.interval)

    def start(self):
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

        return {
            "feature_type": feature_type,
            "time_seconds": self.end_time - self.start_time,

            # CPU Metrics
            "cpu_overall_avg": avg(self.cpu_samples),
            "cpu_peak": max(self.cpu_samples) if self.cpu_samples else None,

            "cpu_per_core_last": self.cpu_per_core_samples[-1] if self.cpu_per_core_samples else None,
            "cpu_per_core_avg": (
                [avg(core) for core in zip(*self.cpu_per_core_samples)]
                if self.cpu_per_core_samples else None
            ),

            "cpu_freq_avg": {
                "current": avg([f["current"] for f in self.cpu_freq_samples]) if self.cpu_freq_samples else None,
                "min": avg([f["min"] for f in self.cpu_freq_samples]) if self.cpu_freq_samples else None,
                "max": avg([f["max"] for f in self.cpu_freq_samples]) if self.cpu_freq_samples else None,
            },

            "cpu_load_avg": {
                "1m": avg([x["1m"] for x in self.cpu_load_samples]),
                "5m": avg([x["5m"] for x in self.cpu_load_samples]),
                "15m": avg([x["15m"] for x in self.cpu_load_samples])
            } if self.cpu_load_samples else None,

            "cpu_times_last": self.cpu_times_samples[-1] if self.cpu_times_samples else None,

            "thread_count_last": self.thread_count_samples[-1] if self.thread_count_samples else None,

            # RAM
            "ram_avg": avg(self.ram_samples),
            "ram_peak": max(self.ram_samples) if self.ram_samples else None,

            # # GPU
            # "gpu_available": self.gpu_handle is not None,
            # "gpu_util_avg": avg(self.gpu_util_samples),
            # "gpu_mem_avg": avg(self.gpu_mem_samples),
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