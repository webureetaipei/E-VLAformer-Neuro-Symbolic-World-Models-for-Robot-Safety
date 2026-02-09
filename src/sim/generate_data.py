from omni.isaac.kit import SimulationApp
# Headless mode is faster but requires specific rendering calls
simulation_app = SimulationApp({"headless": True}) 

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
import omni.replicator.core as rep
import h5py
import numpy as np
import json
import os

# 1. BATCH CONFIGURATION
FILE_ID = "003" 
OUTPUT_DIR = r"C:\Users\Michael\evlaformer_lab\data\output\batch_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"sim_data_batch_{FILE_ID}.hdf5")

world = World(stage_units_in_meters=1.0)

def setup_scene():
    # Light & Cubes
    rep.create.light(light_type="Sphere", intensity=1e6, position=(5, 5, 10), name="Light")
    prim_utils.create_prim("/World/CubeA", "Cube", position=np.array([0.0, 0.0, 0.5]), scale=np.array([0.5, 0.5, 0.5]))
    prim_utils.create_prim("/World/CubeB", "Cube", position=np.array([2.0, 0.0, 0.5]), scale=np.array([0.5, 0.5, 0.5]))
    
    cube_a = XFormPrim("/World/CubeA")
    cube_b = XFormPrim("/World/CubeB")
    
    # --- CAMERA & ANNOTATOR SETUP ---
    camera = rep.create.camera(position=(7, 7, 7), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, resolution=(224, 224))
    
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(rp)
    
    return cube_a, cube_b, rgb_annotator

cube_a, cube_b, rgb_annotator = setup_scene()

# 2. RENDERER WARM-UP (Task 10 stability)
# Forces the engine to load textures and initialize buffers
print("⌛ Warming up renderer...")
for _ in range(10):
    world.step(render=True)

# 3. DATA GENERATION LOOP
print(f"🚀 Generating Real RGB Data for Batch {FILE_ID}...")

rgb_frames = []
collision_events = []
metadata_stream = []
mass_a, mass_b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)

for i in range(20):
    # Orchestrator step is critical for headless rendering
    # rt_subframes=4 ensures materials are loaded and reduces noise
    rep.orchestrator.step(rt_subframes=4)
    
    # Capture RGB Data
    raw_rgb = rgb_annotator.get_data()
    
    # --- ROBUST INDEXING FIX ---
    # Only index if we have a 3D array (Height, Width, Channels)
    if hasattr(raw_rgb, "shape") and len(raw_rgb.shape) == 3:
        rgb_data = raw_rgb[:, :, :3].astype(np.uint8)
        rgb_frames.append(rgb_data)
    else:
        print(f"⚠️ Warning: Frame {i} buffer empty. Using black placeholder.")
        rgb_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Physics/Causal Logic
    pos_a = cube_a.get_world_pose()[0]
    pos_b = cube_b.get_world_pose()[0]
    distance = np.linalg.norm(pos_a - pos_b)
    collision = bool(distance < 1.1)
    
    collision_events.append(collision)
    metadata_stream.append(json.dumps({"frame": i, "mass_a": mass_a, "collision": collision}))

# Save to HDF5
with h5py.File(OUTPUT_PATH, 'w') as f:
    f.create_dataset("rgb", data=np.array(rgb_frames), compression="gzip")
    f.create_dataset("collision_event", data=np.array(collision_events))
    f.create_dataset("metadata", data=np.array(metadata_stream, dtype=h5py.special_dtype(vlen=str)))

print(f"✅ Success! Data saved to {OUTPUT_PATH}")
simulation_app.close()