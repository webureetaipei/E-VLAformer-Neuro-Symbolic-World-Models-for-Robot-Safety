import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# Set file paths for Task 08 randomized data
HDF5_PATH = r"C:\Users\Michael\evlaformer_lab\data\output\randomized_data.hdf5"
SAVE_PATH = r"C:\Users\Michael\evlaformer_lab\docs\images\randomization_validation.png"

# Ensure the docs/images directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def visualize_randomized_collision():
    """Scans HDF5 for a collision event and saves visual proof with DR metadata."""
    if not os.path.exists(HDF5_PATH):
        print(f"❌ Error: HDF5 file not found at {HDF5_PATH}. Run Task 08 generate_data.py first.")
        return

    with h5py.File(HDF5_PATH, 'r') as f:
        # Scan for the first frame where collision_event is True
        collision_flags = f['collision_event'][:]
        collision_frames = np.where(collision_flags == True)[0]

        if len(collision_frames) == 0:
            print("⚠️ No collision detected. Using middle frame for visualization.")
            target_idx = len(collision_flags) // 2
        else:
            target_idx = collision_frames[0]
            print(f"💥 Collision detected at Frame {target_idx}!")

        # Extract RGB and Metadata
        rgb = f['rgb'][target_idx]
        metadata = f['metadata'][target_idx]
        
        # Create professional plot
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        
        ax.imshow(rgb)
        ax.set_title(f"Task 08: Domain Randomization Validation\nImpact Frame: {target_idx}", 
                     fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add a text box with the full randomized metadata
        plt.figtext(0.5, 0.05, f"Captured Metadata:\n{metadata}", 
                    wrap=True, horizontalalignment='center', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Save and Show
        plt.tight_layout()
        plt.savefig(SAVE_PATH, bbox_inches='tight')
        print(f"✅ Randomization validation image saved to: {SAVE_PATH}")
        plt.show()

if __name__ == "__main__":
    visualize_randomized_collision()