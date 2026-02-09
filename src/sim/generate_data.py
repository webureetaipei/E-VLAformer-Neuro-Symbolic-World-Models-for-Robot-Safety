from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) 

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim  # <-- FIXED: Added this import
import omni.replicator.core as rep
import h5py
import numpy as np
import json
import os
import sys

# MANUAL FILENAME: Change this "001" to "002", "003", etc. for each run
FILE_ID = "003" 
OUTPUT_PATH = f"C:/Users/Michael/evlaformer_lab/data/output/batch_v1/sim_data_batch_{FILE_ID}.hdf5"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

world = World(stage_units_in_meters=1.0)

def setup_scene():
    light = rep.create.light(light_type="Sphere", intensity=1e6, position=(0, 0, 5), name="Light")
    
    # Create the raw prims
    prim_utils.create_prim("/World/CubeA", "Cube", position=np.array([0.0, 0.0, 0.5]), scale=np.array([0.5, 0.5, 0.5]))
    prim_utils.create_prim("/World/CubeB", "Cube", position=np.array([2.0, 0.0, 0.5]), scale=np.array([0.5, 0.5, 0.5]))
    
    # Wrap them in XFormPrim to enable physics/pose functions
    cube_a = XFormPrim("/World/CubeA")
    cube_b = XFormPrim("/World/CubeB")
    return cube_a, cube_b

cube_a, cube_b = setup_scene()

print(f"🚀 Generating Data for Batch {FILE_ID}...")

rgb_frames = []
collision_events = []
metadata_stream = []
mass_a, mass_b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)

for i in range(20):
    world.step(render=True)
    pos_a = cube_a.get_world_pose()[0]
    pos_b = cube_b.get_world_pose()[0]
    distance = np.linalg.norm(pos_a - pos_b)
    collision = bool(distance < 1.1)
    
    rgb_frames.append(np.zeros((224, 224, 3), dtype=np.uint8)) 
    collision_events.append(collision)
    metadata_stream.append(json.dumps({"frame": i, "mass_a": mass_a, "collision": collision}))

with h5py.File(OUTPUT_PATH, 'w') as f:
    f.create_dataset("rgb", data=np.array(rgb_frames), compression="gzip")
    f.create_dataset("collision_event", data=np.array(collision_events))
    f.create_dataset("metadata", data=np.array(metadata_stream, dtype=h5py.special_dtype(vlen=str)))

print(f"✅ Success! Saved to {OUTPUT_PATH}")
simulation_app.close()