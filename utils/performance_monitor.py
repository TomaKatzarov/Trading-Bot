import time
import psutil
import torch

class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.gpu_available = torch.cuda.is_available()
    
    def get_system_stats(self):
        return {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "gpu_usage": torch.cuda.memory_allocated() if self.gpu_available else 0,
            "uptime": time.time() - self.start_time
        }
    
    def log_performance(self):
        stats = self.get_system_stats()
        return f"[PERF] CPU: {stats['cpu_usage']}% | RAM: {stats['ram_usage']}% | GPU: {stats['gpu_usage']/1e6:.1f}MB | Uptime: {stats['uptime']:.1f}s"
