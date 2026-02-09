import isaacsim
import os
import h5py
import numpy as np
from omni.isaac.kit import SimulationApp

# 1. Initialization
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World

OUTPUT_FILE = r"C:\Users\Michael\evlaformer_lab\data\output\collision_data.hdf5"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def run_task07():
    world = World() 
    num_frames = 20
    height, width = 512, 512
    
    # 2. Replicator Camera & Annotator Setup (Fixed: Added this section)
    camera = rep.create.camera(position=(10, 10, 10), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (width, height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach(rp)

    # 3. Setup HDF5
    f = h5py.File(OUTPUT_FILE, 'w')
    f.create_dataset("rgb", (num_frames, height, width, 3), dtype='uint8')
    f.create_dataset("collision_event", (num_frames,), dtype='bool')
    f.create_dataset("metadata", (num_frames,), dtype=h5py.string_dtype())

    # 4. Create World Objects
    prim_utils.create_prim("/World/CubeA", "Cube", position=(0, -2, 2))
    prim_utils.create_prim("/World/CubeB", "Cube", position=(0, 2, 2))
    rep.create.light(light_type="dome", intensity=1000)
    
    print(f"🚀 TASK 07: Starting Collision Simulation...", flush=True)

    for i in range(num_frames):
        # Apply motion
        curr_pos_a = prim_utils.get_prim_at_path("/World/CubeA").GetAttribute("xformOp:translate").Get()
        curr_pos_b = prim_utils.get_prim_at_path("/World/CubeB").GetAttribute("xformOp:translate").Get()
        
        # Move them toward each other (0.2 units per frame)
        prim_utils.set_prim_attribute_value("/World/CubeA", "xformOp:translate", curr_pos_a + (0, 0.2, 0))
        prim_utils.set_prim_attribute_value("/World/CubeB", "xformOp:translate", curr_pos_b - (0, 0.2, 0))
        
        # Step Physics and Renderer
        world.step(render=True)
        rep.orchestrator.step() 
        
        # Capture Camera Data (Fixed: Added data retrieval)
        rgb_data = rgb_annot.get_data()
        if rgb_data is not None:
            f["rgb"][i] = rgb_data[:, :, :3]
        
        # Simple proximity check for 'Collision' label
        dist = np.linalg.norm(np.array(curr_pos_a) - np.array(curr_pos_b))
        is_colliding = dist < 2.0 # Adjusted threshold (cube size is usually 2x2x2 by default)
        
        f["collision_event"][i] = is_colliding
        f["metadata"][i] = str({"frame": i, "dist": float(dist), "event": "collision" if is_colliding else "none"})
        
        print(f"   Frame {i}: Distance {dist:.2f} | Colliding: {is_colliding}")

    f.close()
    print(f"✅ TASK 07 SUCCESS: Causal data saved to {OUTPUT_FILE}")
    simulation_app.close()

if __name__ == "__main__":
    run_task07()