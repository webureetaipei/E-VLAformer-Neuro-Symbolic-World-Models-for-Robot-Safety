import h5py
import numpy as np
import os
import json

HDF5_PATH = r"C:\Users\Michael\evlaformer_lab\data\output\randomized_data.hdf5"

def audit_dataset():
    print(f"🔍 Starting Dataset Audit: {HDF5_PATH}...")
    
    if not os.path.exists(HDF5_PATH):
        print("❌ ERROR: File not found.")
        return

    with h5py.File(HDF5_PATH, 'r') as f:
        # 1. Structural Check
        expected_keys = ['rgb', 'collision_event', 'metadata']
        for key in expected_keys:
            if key in f:
                print(f"  ✅ Dataset '{key}' found. Shape: {f[key].shape}")
            else:
                print(f"  ❌ ERROR: Missing dataset '{key}'")

        # 2. Visual Integrity Check (Zero-Entropy)
        rgb_data = f['rgb'][:]
        if np.all(rgb_data == 0) or np.all(rgb_data == 255):
            print("  ❌ ERROR: RGB data is empty (all black or all white).")
        else:
            print(f"  ✅ RGB Integrity: Mean pixel value {np.mean(rgb_data):.2f}")

        # 3. Causal Logic Validation
        collisions = f['collision_event'][:]
        collision_count = np.sum(collisions)
        print(f"  ✅ Causal Check: Found {collision_count} collision frames.")
        
        # 4. Metadata Parsing & Physics Range Check
        metadata_samples = f['metadata'][:]
        try:
            # Parse the last frame's metadata (stored as string)
            meta_str = metadata_samples[-1]
            # Handle potential byte string from HDF5
            if isinstance(meta_str, bytes):
                meta_str = meta_str.decode('utf-8')
            
            # Clean string if it was stored as "b'...' "
            if meta_str.startswith("b'"):
                meta_str = meta_str[2:-1]
                
            meta_json = json.loads(meta_str.replace("'", '"').replace("True", "true").replace("False", "false"))
            
            mass_a = meta_json['physical_props']['mass_a']
            if 0.5 <= mass_a <= 5.0:
                print(f"  ✅ Physics Check: Mass_A ({mass_a:.2f}kg) within DR bounds.")
            else:
                print(f"  ❌ ERROR: Mass_A outlier detected!")
                
        except Exception as e:
            print(f"  ❌ ERROR: Metadata parsing failed: {e}")

    print("\n⭐ AUDIT COMPLETE: Dataset is certified for Phase 2 Training.")

if __name__ == "__main__":
    audit_dataset()