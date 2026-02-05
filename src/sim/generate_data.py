# src/sim/generate_data.py
import isaacsim
import os
import h5py
import numpy as np
from omni.isaac.kit import SimulationApp

# 1. Initialization
config = {"headless": True, "active_gpu": 0}
simulation_app = SimulationApp(config)

# 2. Replicator Deferred Import
import omni.replicator.core as rep

OUTPUT_FILE = r"C:\Users\Michael\evlaformer_lab\data\output\dataset_v1.hdf5"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def setup_hdf5(num_frames, height, width):
    """Pre-allocates HDF5 datasets for optimized I/O performance."""
    f = h5py.File(OUTPUT_FILE, 'w')
    
    # Create datasets (RGB: uint8, Segmentation: uint8)
    f.create_dataset("rgb", (num_frames, height, width, 3), dtype='uint8', compression="gzip")
    f.create_dataset("semantic", (num_frames, height, width), dtype='uint8', compression="gzip")
    
    # Physics data and semantic descriptions (JSON Strings)
    f.create_dataset("metadata", (num_frames,), dtype=h5py.string_dtype(encoding='utf-8'))
    return f

def run_task06():
    num_frames = 10
    height, width = 512, 512
    hdf5_file = setup_hdf5(num_frames, height, width)

    with rep.new_layer():
        rep.create.light(light_type="dome", intensity=1000)
        
        # Object diversity generation
        cube = rep.create.cube(semantics=[('class', 'cube')])
        with rep.trigger.on_frame(max_execs=num_frames):
            with cube:
                rep.modify.pose(position=rep.distribution.uniform((-5,-5,5),(5,5,10)))
        
        # Attach Annotators (Manual data extraction without Writer)
        camera = rep.create.camera(position=(15, 15, 15), look_at=(0, 0, 0))
        rp = rep.create.render_product(camera, (width, height))
        
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(rp)
        
        sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        sem_annot.attach(rp)

    print(f"🚀 TASK 06: Logging to HDF5 -> {OUTPUT_FILE}", flush=True)

    for i in range(num_frames):
        print(f"   Step {i+1}/{num_frames}...", flush=True)
        rep.orchestrator.step()
        
        # Core: Retrieve Annotator data
        rgb_data = rgb_annot.get_data()
        sem_data = sem_annot.get_data()
        
        if rgb_data is not None:
            # Store into HDF5 (RGB channels only)
            hdf5_file["rgb"][i] = rgb_data[:, :, :3]
            hdf5_file["semantic"][i] = sem_data["data"]
            
            # Store metadata (Foundation for Event Reasoning)
            metadata_entry = {
                "frame": i, 
                "description": "Cube randomly placed in 3D space"
            }
            hdf5_file["metadata"][i] = str(metadata_entry)

    hdf5_file.close()
    print("✅ SUCCESS! HDF5 Dataset Generated.", flush=True)
    simulation_app.close()

if __name__ == "__main__":
    run_task06()