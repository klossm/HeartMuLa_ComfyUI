
import sys
import os
import torch
import time
import csv
import argparse
import gc
import threading

# Add ComfyUI path setups
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

# Mock folder_paths
try:
    import folder_paths
except ImportError:
    class FolderPaths:
        def __init__(self):
            self.models_dir = "models"
            self._paths = {}
        def get_output_directory(self): return "output"
        def get_temp_directory(self): return "temp"
        def add_model_folder_path(self, name, path):
            if name not in self._paths: self._paths[name] = []
            self._paths[name].append(path)
        def get_folder_paths(self, name):
            return self._paths.get(name, [name])
    sys.modules["folder_paths"] = FolderPaths()

# Import Manager
try:
    from __init__ import HeartMuLaModelManager
except ImportError:
    sys.path.append(current_dir)
    from __init__ import HeartMuLaModelManager

class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.data = []
        self.start_time = time.time()

    def run(self):
        while not self.stop_event.is_set():
            current_time = time.time() - self.start_time
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            self.data.append((current_time, allocated, reserved))
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def run_benchmark(mode):
    print(f"--- Starting Benchmark: {mode} (20s audio) ---")
    
    quantize = (mode == "fp4")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Start Monitor
    monitor = MemoryMonitor(interval=0.2)
    monitor.start()
    
    try:
        manager = HeartMuLaModelManager()
        
        print("Loading pipeline...")
        # Assuming '3B' model
        pipe = manager.get_gen_pipeline(version="3B", quantize_4bit=quantize)
        
        # Inputs for 20 seconds of audio
        lyrics = "[Verse]\nTesting with a longer duration\nPushing the memory timeline\nTwenty seconds of generation"
        tags = "electronic, fast, benchmark, long"
        max_audio_length_ms = 20000 
        
        print("Starting generation...")
        
        gen_start = time.time()
        output_path = f"benchmark_{mode}_20s.wav"
        
        with torch.inference_mode():
            pipe(
                inputs={"lyrics": lyrics, "tags": tags},
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=50,
                temperature=1.0,
                cfg_scale=1.5,
                keep_model_loaded=True
            )
        
        gen_end = time.time()
        print(f"Generation completed in {gen_end - gen_start:.2f}s")

    except Exception as e:
        print(f"Generation Failed: {e}")
    finally:
        monitor.stop()
        monitor.join()
    
    # Save Timeline Data
    timeline_file = f"timeline_{mode}_20s.csv"
    with open(timeline_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Allocated VRAM (GB)', 'Reserved VRAM (GB)'])
        writer.writerows(monitor.data)
    
    print(f"Timeline saved to {timeline_file}")
    
    # Calculate stats
    peak_allocated = max(d[1] for d in monitor.data) if monitor.data else 0
    peak_reserved = max(d[2] for d in monitor.data) if monitor.data else 0
    
    print(f"Peak Allocated: {peak_allocated:.2f} GB")
    print(f"Peak Reserved: {peak_reserved:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["no_quant", "fp4"])
    args = parser.parse_args()
    
    run_benchmark(args.mode)
