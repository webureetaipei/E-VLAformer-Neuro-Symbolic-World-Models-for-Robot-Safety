from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

import os
import h5py
import random
import numpy as np
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils

def apply_occlusion_blink(cube_prim_path_str, occlusion_probability=0.10):
    # Fix: Use the direct USD Stage API to get the prim
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(cube_prim_path_str)
    
    if not prim or not prim.IsValid():
        return None

    blink = random.random() < occlusion_probability
    if blink:
        prim_utils.set_prim_visibility(prim, visible=False)
        return True
    else:
        prim_utils.set_prim_visibility(prim, visible=True)
        return False

def generate_hardened_batch(output_path, num_frames=100):
    print("DEBUG: Function started...")
    
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    try:
        # Create the environment
        cube_path = "/World/RedCube"
        prim_utils.create_prim(cube_path, "Cube", position=np.array([0, 0, 0.5]))
        
        # Warmup
        for i in range(60):
            simulation_app.update()
        
        print(f"--- 🚀 Starting Data Generation: {abs_output_path} ---")
        
        with h5py.File(abs_output_path, 'w') as f:
            dset = f.create_dataset("occluded_flag", (num_frames,), dtype='i')
            
            for frame in range(num_frames):
                res = apply_occlusion_blink(cube_path)
                
                if res is None:
                    # Try to wait one more frame if prim is missing
                    simulation_app.update()
                    res = apply_occlusion_blink(cube_path)
                
                dset[frame] = 1 if res else 0
                simulation_app.update()
                
                if frame % 20 == 0:
                    status = "🕶️ BLINK" if res else "👁️ VISIBLE"
                    print(f"Frame {frame:03d}: {status}")

        print("✅ Task 18 Complete. Dataset secured.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR INSIDE GENERATOR: {e}")

if __name__ == "__main__":
    try:
        generate_hardened_batch("data/raw/task18_occlusion_test_001.h5")
    finally:
        print("🛑 Shutting down Simulation App...")
        simulation_app.close()