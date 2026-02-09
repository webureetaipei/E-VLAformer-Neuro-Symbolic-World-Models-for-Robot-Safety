import subprocess
import os
import time

# Use the absolute path to Isaac Sim's python.bat
PYTHON_PATH = r"C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat"
PROJECT_ROOT = r"C:\Users\Michael\evlaformer_lab"
GENERATOR_SCRIPT = os.path.join(PROJECT_ROOT, "src", "sim", "generate_data.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "batch_v1")
NUM_BATCHES = 5 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_batch():
    print(f"🚀 Starting Batch Data Generation (Total: {NUM_BATCHES})...")
    
    for i in range(NUM_BATCHES):
        start_time = time.time()
        filename = os.path.join(OUTPUT_DIR, f"sim_data_batch_{i:03d}.hdf5")
        
        print(f"📦 Generating Batch {i+1}/{NUM_BATCHES} -> {filename}")
        
        # WE USE THE BAT FILE DIRECTLY with shell=True
        # This forces Windows to run the BAT logic which sets up the omni paths
        command = f'"{PYTHON_PATH}" "{GENERATOR_SCRIPT}" --output "{filename}"'
        
        result = subprocess.run(
            command, 
            cwd=PROJECT_ROOT,
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"   ✅ Success! Time: {duration:.2f}s")
        else:
            print(f"   ❌ Failed Batch {i+1}!")
            # This will show us if it's still an import error or something else
            print(f"   Error Log: {result.stderr}")

    print(f"\n⭐ BATCH GENERATION COMPLETE. Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch()