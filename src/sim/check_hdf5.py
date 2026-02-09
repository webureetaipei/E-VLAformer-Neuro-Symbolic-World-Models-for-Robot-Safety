import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Path Configuration
BATCH_DIR = r"C:\Users\Michael\evlaformer_lab\data\output\batch_v1"
REPORT_DIR = r"C:\Users\Michael\evlaformer_lab\docs\images"
os.makedirs(REPORT_DIR, exist_ok=True)

def visualize_batch_diversity():
    # Find all hdf5 files in the batch directory
    files = [f for f in os.listdir(BATCH_DIR) if f.endswith('.hdf5')]
    files.sort() # Ensure they are in order (001, 002, 003)
    
    if not files:
        print(f"❌ No HDF5 files found in {BATCH_DIR}")
        return

    num_files = len(files)
    fig, axes = plt.subplots(1, num_files, figsize=(15, 5))
    
    # Handle the case where there's only 1 file
    if num_files == 1:
        axes = [axes]

    print(f"🔍 Analyzing {num_files} batches for diversity...")

    for i, filename in enumerate(files):
        file_path = os.path.join(BATCH_DIR, filename)
        
        with h5py.File(file_path, 'r') as f:
            # 1. Extract Data
            # We take frame index 10 (middle of the 20-frame sequence)
            sample_idx = 10
            rgb = f['rgb'][sample_idx]
            collision = f['collision_event'][sample_idx]
            
            # 2. Extract Metadata for the label
            raw_meta = f['metadata'][sample_idx]
            meta = json.loads(raw_meta)
            
            # 3. Plotting
            axes[i].imshow(rgb)
            axes[i].set_title(f"Batch: {filename[-8:-5]}\nColliding: {collision}")
            axes[i].axis('off')
            
            # Print stats to terminal
            print(f"✅ Processed {filename}: Mass_A={meta.get('mass_a', 'N/A'):.2f}kg")

    plt.tight_layout()
    
    # Save the Diversity Grid for README.md
    output_path = os.path.join(REPORT_DIR, "randomization_validation.png")
    plt.savefig(output_path)
    print(f"\n⭐ Diversity Grid saved to: {output_path}")
    
    # Show the plot if running locally
    plt.show()

if __name__ == "__main__":
    visualize_batch_diversity()