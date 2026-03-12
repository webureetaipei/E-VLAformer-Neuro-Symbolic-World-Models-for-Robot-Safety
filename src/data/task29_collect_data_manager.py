import subprocess
import time
import sys
import os

# CONFIGURATION
TOTAL_RUNS = 48 
HARVESTER_PATH = r"C:\Users\Michael\evlaformer_lab\src\data\task29_expert_harvester_randomized.py"
PYTHON_EXE = sys.executable 

def run_collection():
    print(f"🌟 Starting Stability Phase: {TOTAL_RUNS} Episodes")
    success_count = 0
    
    for i in range(TOTAL_RUNS):
        print(f"\n--- 🏁 Attempting Episode {i+1} / {TOTAL_RUNS} ---")
        
        start_time = time.time()
        # Launch the harvester
        result = subprocess.run([PYTHON_EXE, HARVESTER_PATH])
        elapsed_time = time.time() - start_time

        # 🛑 CIRCUIT BREAKER LOGIC
        # If the simulator crashes in less than 10 seconds, it's a Fatal Error.
        if elapsed_time < 10:
            print("\n🚨 CRITICAL FAILURE: Isaac Sim crashed instantly.")
            print("This is likely a Path, DLL, or Python environment error.")
            print("Stopping the manager to prevent an infinite loop.")
            break 

        if result.returncode == 0:
            success_count += 1
            print(f"✅ Success! ({success_count}/{TOTAL_RUNS})")
            time.sleep(2)
        else:
            print(f"❌ Harvester failed (Code: {result.returncode}). Cooling down...")
            time.sleep(15)

    print(f"\n🎉 Process Stopped. Total collected this session: {success_count}")

if __name__ == "__main__":
    run_collection()