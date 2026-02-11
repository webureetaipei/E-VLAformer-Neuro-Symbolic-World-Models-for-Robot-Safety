# 1. Initialize SimulationApp FIRST to load Replicator modules
from omni.isaac.kit import SimulationApp

# Configuration for headless mode
CONFIG = {"headless": True}
simulation_app = SimulationApp(CONFIG)

# 2. Now import Replicator and Isaac core modules
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
import h5py
import numpy as np
import os

def setup_scene():
    """Sets up a basic ground plane and lighting."""
    with rep.new_layer():
        # Create a basic ground plane so the camera isn't looking at a void
        rep.create.plane(scale=10, position=(0, 0, 0))
        
        # Studio Lighting: Dome for ambient and Distant for shadows
        rep.create.light(light_type="dome", intensity=800)
        rep.create.light(light_type="distant", intensity=1500, rotation=(45, 45, 0))

def generate_randomized_batch(num_frames=20):
    # Initialize the high-level Isaac World
    world = World(stage_units_in_meters=1.0)
    setup_scene()

    # 3. Setup Camera: Positioned to see the center of the stage
    # (2, 2, 2) is far enough to see the robot/objects without clipping
    camera = rep.create.camera(position=(2.0, 2.0, 2.0), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (512, 512))
    
    # Initialize and attach the RGB annotator
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach(rp)

    # 4. Start Simulation and Warm Up
    world.play()
    print("⏳ Warming up RTX renderer (60 steps)...")
    for _ in range(60):
        world.step(render=True)
    
    # 5. Data Capture Loop
    os.makedirs('data/output', exist_ok=True)
    output_path = 'data/output/randomized_data.hdf5'

    with h5py.File(output_path, 'w') as f:
        rgb_ds = f.create_dataset("rgb", (num_frames, 512, 512, 3), dtype='uint8')
        
        print(f"🚀 Capturing {num_frames} frames...")
        
        for i in range(num_frames):
            # Step the orchestrator (subframes ensure high-quality light)
            rep.orchestrator.step(rt_subframes=12) 
            
            # Fetch data from GPU buffer
            data = rgb_annot.get_data()
            
            if data is not None:
                # Remove alpha channel and save
                rgb_ds[i] = data[:, :, :3]
                mean_val = np.mean(rgb_ds[i])
                print(f"✅ Frame {i} | Mean Brightness: {mean_val:.2f}")
                
                # Check for absolute black
                if mean_val < 0.1:
                    print("⚠️ Warning: Image is still very dark. Check camera/lights.")
            else:
                print(f"❌ Frame {i} failed to capture data.")

    # 6. Cleanup
    world.stop()
    simulation_app.close()
    print(f"🎉 Process Complete. File saved to {output_path}")

if __name__ == "__main__":
    generate_randomized_batch()