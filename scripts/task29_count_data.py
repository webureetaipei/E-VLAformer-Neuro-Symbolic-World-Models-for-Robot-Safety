import os
from collections import Counter

def count_episode_types():
    data_dir = r"C:\Users\Michael\evlaformer_lab\data"
    
    # Find all .h5 files in the directory
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"❌ No .h5 files found in {data_dir}!")
        return

    # Count the occurrences of each type based on the filename
    counts = Counter()
    for filename in h5_files:
        # Filename format: task29_TYPE_timestamp.h5
        parts = filename.split('_')
        if len(parts) >= 2:
            episode_type = parts[1]
            counts[episode_type] += 1

    # Print the results
    print(f"\n🎉 Total HDF5 files collected: {len(h5_files)}")
    print("-" * 30)
    print(f"🟢 NORMAL:       {counts.get('NORMAL', 0)}")
    print(f"🧱 OCCLUSION:    {counts.get('OCCLUSION', 0)}")
    print(f"🎱 PERTURBATION: {counts.get('PERTURBATION', 0)}")
    print("-" * 30)
    
    # Quick sanity check
    expected = len(h5_files)
    actual = counts.get('NORMAL', 0) + counts.get('OCCLUSION', 0) + counts.get('PERTURBATION', 0)
    if expected != actual:
        print(f"⚠️ Warning: Found {expected - actual} files with an unrecognized naming format.")

if __name__ == "__main__":
    count_episode_types()