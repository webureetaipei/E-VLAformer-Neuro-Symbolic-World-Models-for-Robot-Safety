import os
import h5py
import numpy as np
import glob
from PIL import Image

def extract_h5_images():
    # Path where your HDF5 files are stored
    data_dir = r"C:\Users\Michael\evlaformer_lab\data"
    
    # Path where the extracted PNG images will be saved
    output_dir = r"C:\Users\Michael\evlaformer_lab\debug_images"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 🔥 NEW FEATURE: Clear old debug images before each run
    # ==========================================
    old_images = glob.glob(os.path.join(output_dir, "*.png"))
    for f in old_images:
        os.remove(f)
    if old_images:
        print(f"🧹 Cleared {len(old_images)} old debug images.")

    # Find all .h5 files in the directory
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"❌ No .h5 files found in {data_dir}!")
        return

    print(f"🔍 Found {len(h5_files)} HDF5 files. Starting image extraction...\n")

    for filename in h5_files:
        filepath = os.path.join(data_dir, filename)
        
        try:
            with h5py.File(filepath, 'r') as f:
                data_group = f['data']
                steps = list(data_group.keys())
                
                # Sort the steps numerically (so step_2 comes before step_10)
                steps.sort(key=lambda x: int(x.split('_')[1]))
                
                if not steps:
                    print(f"⚠️ Skipping {filename}: No step data found inside.")
                    continue
                
                # Extract 3 key frames: Start, Mid, and End
                start_step = steps[0]
                mid_step = steps[len(steps) // 2]
                end_step = steps[-1]
                
                frames_to_extract = {
                    "start": start_step,
                    "mid": mid_step,
                    "end": end_step
                }
                
                for label, step_name in frames_to_extract.items():
                    # Read the image numpy array
                    img_array = data_group[step_name]['obs/image'][:]
                    
                    # Convert the numpy array to a PIL Image
                    img = Image.fromarray(img_array)
                    
                    # Format the output filename: e.g., task29_OCCLUSION_1700000_mid.png
                    base_name = filename.replace('.h5', '')
                    out_name = f"{base_name}_{label}.png"
                    out_path = os.path.join(output_dir, out_name)
                    
                    # Save the image
                    img.save(out_path)
                    
            print(f"✅ Successfully extracted: {filename} -> Saved start, mid, end images.")
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    print(f"\n🎉 Inspection complete! Please open the '{output_dir}' folder to check the PNG images.")

if __name__ == "__main__":
    extract_h5_images()