# runner.py

import os
import shutil
import subprocess
import time
from datetime import datetime

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "log.txt")

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, also_print=True):
    """
    Writes a message to the log file and optionally prints it.
    Ensures that the log directory exists.
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    line = f"[{timestamp()}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line)

def prepare_results_dir():
    """
    Clears and prepares the results directory.
    """
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        print(f"[{timestamp()}] Cleared old results in '{RESULTS_DIR}/'")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Log started at {timestamp()}\n\n")
    log(f"Created clean directory '{RESULTS_DIR}/'")

def run_script(script_path):
    """
    Runs a Python script, measures execution time, logs stdout/stderr.
    """
    log(f"Running script: {script_path}")
    start_time = time.time()
    
    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    log(f"Completed: {script_path} (elapsed time: {elapsed:.2f} s)")
    
    separator = "─" * 60
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n[stderr]\n")
            f.write(result.stderr)
        f.write(f"\n{separator}\n")
    
    print(result.stdout)
    if result.stderr:
        print("[stderr]\n", result.stderr)

def main():
    prepare_results_dir()
    log("=== Starting cognitive entropy analysis ===")

    run_script("analyze_entropy_landauer.py")
    run_script("plot_discrimination_vs_landauer_v3_annotated.py")

    log("=== All tasks completed. Results saved in 'results/' ===")

if __name__ == "__main__":
    main()
