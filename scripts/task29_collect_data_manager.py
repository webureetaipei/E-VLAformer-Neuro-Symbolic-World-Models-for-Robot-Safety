import subprocess
import time
import sys
import os

# CONFIGURATION
TOTAL_EPISODES = 100
HARVESTER_PATH = r"C:\Users\Michael\evlaformer_lab\src\data\task29_expert_harvester_randomized.py"

# 🔥 FIX: Automatically use the exact same Isaac Sim Python you used to launch the manager
PYTHON_EXE = sys.executable 

def run_collection():
    print(f"🌟 Starting Mass Data Collection: {TOTAL_EPISODES} Episodes (HEADLESS MODE)")
    
    success_count = 0
    for i in range(TOTAL_EPISODES):
        print(f"\n--- 🏁 Processing Episode {i+1}/{TOTAL_EPISODES} ---")
        
        try:
            # check=True means it will throw an error if the simulation crashes
            result = subprocess.run([PYTHON_EXE, HARVESTER_PATH], check=True)
            if result.returncode == 0:
                success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Episode {i+1} failed. The simulation might have crashed. Moving to next...")
        except KeyboardInterrupt:
            print("\n🛑 Collection stopped by user.")
            break
            
        # Give your GPU 2 seconds to clear memory before starting the next episode
        time.sleep(2)

    print(f"\n🎉 Finished! Successfully collected {success_count} episodes.")

if __name__ == "__main__":
    run_collection()