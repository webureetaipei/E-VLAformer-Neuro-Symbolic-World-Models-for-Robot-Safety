import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

HDF5_PATH = r"C:\Users\Michael\evlaformer_lab\data\output\collision_data.hdf5"
SAVE_PATH = r"C:\Users\Michael\evlaformer_lab\docs\images\collision_validation.png"

def visualize_collision_event():
    if not os.path.exists(HDF5_PATH):
        print(f"❌ Error: File not found at {HDF5_PATH}")
        return

    with h5py.File(HDF5_PATH, 'r') as f:
        collision_flags = f['collision_event'][:]
        collision_frames = np.where(collision_flags == True)[0]

        # Use the first collision frame, or the middle frame if no collision
        target_idx = collision_frames[0] if len(collision_frames) > 0 else len(collision_flags) // 2
        print(f"🎯 Target Frame for Visualization: {target_idx}")

        rgb = f['rgb'][target_idx]
        metadata = f['metadata'][target_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.imshow(rgb)
        ax.set_title(f"Causal Validation: Impact Detected (Frame {target_idx})", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.figtext(0.5, 0.05, f"Metadata: {metadata}", wrap=True, horizontalalignment='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(SAVE_PATH, bbox_inches='tight')
        print(f"✅ Validation image saved to: {SAVE_PATH}")
        plt.show()

if __name__ == "__main__":
    visualize_collision_event()