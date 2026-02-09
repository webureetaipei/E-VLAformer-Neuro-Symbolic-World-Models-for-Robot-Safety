import isaacsim
import os
import h5py
import numpy as np
import random
from omni.isaac.kit import SimulationApp

# 1. Initialize Isaac Sim
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core import World

OUTPUT_FILE = r"C:\Users\Michael\evlaformer_lab\data\output\randomized_data.hdf5"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def run_task08():
    world = World()
    num_frames = 20
    height, width = 512, 512

    # Camera & Annotator Setup
    camera = rep.create.camera(position=(12, 12, 12), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (width, height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach(rp)

    # Use 'with' for safe HDF5 handling
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset("rgb", (num_frames, height, width, 3), dtype='uint8')
        f.create_dataset("collision_event", (num_frames,), dtype='bool')
        f.create_dataset("metadata", (num_frames,), dtype=h5py.string_dtype())

        # Create Objects & Light
        rep.create.light(light_type="dome", intensity=rep.distribution.uniform(800, 2000))
        cube_a = rep.create.cube(semantics=[('class', 'cube_A')])
        cube_b = rep.create.cube(semantics=[('class', 'cube_B')])

        mass_a = random.uniform(0.5, 5.0)
        mass_b = random.uniform(0.5, 5.0)

        print(f"🚀 TASK 08: Starting Randomized Simulation...", flush=True)

        for i in range(num_frames):
            pos_a = (0, -3 + (i * 0.25), 1)
            pos_b = (0, 3 - (i * 0.25), 1)
            
            # FIXED: Wrapped color distribution in a list to match VtArray<GfVec3f>
            with cube_a:
                rep.modify.pose(position=pos_a)
                rep.modify.attribute("primvars:displayColor", rep.distribution.uniform([(0,0,0)], [(1,1,1)]))
            with cube_b:
                rep.modify.pose(position=pos_b)
                rep.modify.attribute("primvars:displayColor", rep.distribution.uniform([(0,0,0)], [(1,1,1)]))

            world.step(render=True)
            rep.orchestrator.step() 
            
            rgb_data = rgb_annot.get_data()
            if rgb_data is not None:
                f["rgb"][i] = rgb_data[:, :, :3]
            
            dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
            is_colliding = dist < 2.0 
            
            metadata_entry = {
                "frame": i,
                "collision": is_colliding,
                "physical_props": {"mass_a": mass_a, "mass_b": mass_b}
            }
            f["collision_event"][i] = is_colliding
            f["metadata"][i] = str(metadata_entry)
            print(f"   Step {i+1}/{num_frames} completed...", flush=True)

    print(f"✅ SUCCESS: Randomized Dataset Generated at {OUTPUT_FILE}")
    simulation_app.close()

if __name__ == "__main__":
    run_task08()