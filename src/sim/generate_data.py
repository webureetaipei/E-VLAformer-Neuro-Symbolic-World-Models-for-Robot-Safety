# src/sim/generate_data.py
import isaacsim
import os
import time
from omni.isaac.kit import SimulationApp

# 1. Setup Configuration
config = {
    "headless": True,
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
    "renderer": "RayTracedLighting",
    "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]
}

print("⏳ Initializing Isaac Sim engine...", flush=True)
simulation_app = SimulationApp(config)

# Ensure Absolute Path for Windows
OUTPUT_DIR = r"C:\Users\Michael\evlaformer_lab\data\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_data():
    # Late import to avoid extension initialization conflicts
    import omni.replicator.core as rep 
    
    print(f"🚀 Data Generation Started. Target: {OUTPUT_DIR}", flush=True)
    
    with rep.new_layer():
        # Lighting & Scene
        rep.create.light(light_type="dome", intensity=1000)
        rep.create.plane(scale=10, visible=True)
        cube = rep.create.cube(position=(0, 0, 10), scale=2.0, semantics=[('class', 'cube')])
        
        # Define the randomization logic
        with rep.trigger.on_frame(max_execs=5): 
            with cube:
                rep.modify.pose(
                    position=rep.distribution.uniform((-5, -5, 5), (5, 5, 15)),
                    rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
                )
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

        # Camera & Writer
        camera = rep.create.camera(position=(20, 20, 20), look_at=(0, 0, 0))
        render_product = rep.create.render_product(camera, (512, 512))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=OUTPUT_DIR, rgb=True, semantic_segmentation=True)
        writer.attach(render_product)

    # --- THE ROBUST STEPPING FIX ---
    print("📸 Rendering 5 frames with manual GPU sync...", flush=True)
    
    # Instead of 'run()', we step manually. 
    # This prevents the "Unexpected Keyword" or "No Attribute" errors.
    for i in range(5):
        print(f"   Step {i+1}/5...", flush=True)
        rep.orchestrator.step()
        # Give the app a few ticks to handle internal I/O after each step
        for _ in range(10):
            simulation_app.update()

    # Final "Flush" to ensure Windows writes the files to disk
    print("💾 Finalizing file I/O...", flush=True)
    for _ in range(100):
        simulation_app.update()

    print("✅ SUCCESS! Check your folder now.", flush=True)

if __name__ == "__main__":
    try:
        generate_synthetic_data()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🛑 System Exit.")
        # Ensure cleanup and hard exit to prevent access violation
        simulation_app.close()
        time.sleep(1)
        import os
        os._exit(0)