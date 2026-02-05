import h5py
import matplotlib.pyplot as plt
import os

# Set file paths
HDF5_PATH = r"C:\Users\Michael\evlaformer_lab\data\output\dataset_v1.hdf5"
SAVE_PATH = r"C:\Users\Michael\evlaformer_lab\docs\images\visual_validation.png"

# Ensure the docs/images directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def visualize_standard_config():
    """Reads data from HDF5 and creates a side-by-side RGB vs Semantic Mask plot."""
    with h5py.File(HDF5_PATH, 'r') as f:
        # Read the first frame (index 0)
        rgb = f['rgb'][0]
        semantic = f['semantic'][0]
        
        # Create side-by-side plot (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # Left subplot: RGB Input Image
        axes[0].imshow(rgb)
        axes[0].set_title("Standardized Input: RGB", fontsize=14, fontweight='bold')
        axes[0].axis('off') # Hide coordinate axes
        
        # Right subplot: Color-encoded Semantic Mask
        im = axes[1].imshow(semantic, cmap='viridis')
        axes[1].set_title("Ground Truth: Semantic Mask", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save to docs/images for README documentation
        plt.savefig(SAVE_PATH, bbox_inches='tight')
        print(f"✅ Standardized validation image generated and saved to: {SAVE_PATH}")
        
        # Display the result window
        plt.show()

if __name__ == "__main__":
    if os.path.exists(HDF5_PATH):
        visualize_standard_config()
    else:
        print(f"❌ Error: HDF5 file not found at {HDF5_PATH}. Please run generate_data.py first.")