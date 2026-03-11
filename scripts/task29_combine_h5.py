import h5py
import os
import numpy as np

def combine_dataset():
    # 1. Setup Paths
    data_dir = r"C:\Users\Michael\evlaformer_lab\data"
    output_path = os.path.join(data_dir, "task30_training_master.h5")
    
    # 2. Find all individual episode files
    # We exclude any existing 'master' file to avoid recursive loops
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5') and 'master' not in f]
    h5_files.sort() # Sorting ensures chronological or alphabetical consistency

    if not h5_files:
        print("❌ No H5 files found in the data directory!")
        return

    print(f"🚀 Found {len(h5_files)} episodes. Starting the merge...")

    # 3. Create the Master File
    with h5py.File(output_path, 'w') as master_f:
        master_data_group = master_f.create_group("data")
        total_global_steps = 0
        episode_markers = [] # To keep track of where one episode ends and another begins

        for file_idx, h5_name in enumerate(h5_files):
            file_path = os.path.join(data_dir, h5_name)
            
            try:
                with h5py.File(file_path, 'r') as source_f:
                    source_steps = list(source_f['data'].keys())
                    
                    # Sort steps numerically (step_0, step_10, step_20...)
                    source_steps.sort(key=lambda x: int(x.split('_')[1]))

                    for step_name in source_steps:
                        # Create a continuous global index: step_0, step_1, step_2...
                        new_step_name = f"step_{total_global_steps}"
                        
                        # Efficiently copy the entire step group (obs, action, etc.)
                        source_f.copy(f"data/{step_name}", master_data_group, name=new_step_name)
                        total_global_steps += 1
                
                print(f"✅ [{file_idx+1}/{len(h5_files)}] Merged: {h5_name} ({len(source_steps)} steps)")
            
            except Exception as e:
                print(f"⚠️ Error processing {h5_name}: {e}")

        # 4. Save Meta-Data for Task 30
        master_f.create_dataset("total_steps", data=total_global_steps)
        print(f"\n" + "="*40)
        print(f"🎉 MASTER DATASET COMPLETE!")
        print(f"📍 Location: {output_path}")
        print(f"📊 Total Global Steps: {total_global_steps}")
        print("="*40)

if __name__ == "__main__":
    combine_dataset()