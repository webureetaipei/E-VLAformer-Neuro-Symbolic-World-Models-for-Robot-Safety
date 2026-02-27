import os
import h5py
import numpy as np
import random
from isaacsim import SimulationApp

# 1. Start App (headless=False is required to keep the renderer active)
simulation_app = SimulationApp({"headless": False, "width": "1280", "height": "720"})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from omni.isaac.franka import Franka 
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
import omni.replicator.core as rep

class ScramblerHarvester:
    def __init__(self, h5_path="data/expert_trajectories.h5"):
        self.h5_path = h5_path
        if not os.path.exists("data"): 
            os.makedirs("data")
        
    def randomize_scene(self, cube):
        # Randomize position
        x, y = random.uniform(0.4, 0.6), random.uniform(-0.15, 0.15)
        cube.set_world_pose(position=np.array([x, y, 0.05]))
        
        # Randomize color
        material = cube.get_applied_visual_material()
        if material:
            material.set_color(np.array([random.random(), random.random(), random.random()]))
        return np.array([x, y, 0.05])

    def collect_episode(self, episode_id, world, robot, cube, rgb_annotator):
        obs_pixels, obs_joints = [], []
        controller = RMPFlowController(name="target_follower", robot_articulation=robot)
        target_pos = self.randomize_scene(cube)
        
        print(f"\n--- 🎬 Starting Episode {episode_id} | Target: {target_pos} ---")

        # Give the renderer time to generate shadows and textures
        for _ in range(30):
            rep.orchestrator.step()
            world.step(render=True)

        for step in range(150):
            rep.orchestrator.step()
            
            # ✅ THIS MUST BE TRUE: Forces the GPU to draw the movement!
            world.step(render=True) 
            
            actions = controller.forward(target_end_effector_position=target_pos)
            robot.apply_action(actions)
            
            raw_rgb = rgb_annotator.get_data()
            if raw_rgb is not None and raw_rgb.size > 0:
                # Ultra-safe data conversion
                if raw_rgb.dtype == np.uint8:
                    img_frame = raw_rgb[:, :, :3]
                elif np.max(raw_rgb) <= 1.0:
                    img_frame = (raw_rgb[:, :, :3] * 255).astype(np.uint8)
                else:
                    img_frame = raw_rgb[:, :, :3].astype(np.uint8)
                
                if np.max(img_frame) > 0:
                    obs_pixels.append(np.array(img_frame, copy=True))
                    obs_joints.append(np.array(robot.get_joint_positions(), copy=True))

        if len(obs_pixels) > 0:
            final_images = np.stack(obs_pixels)
            
            # 🔍 --- AUTOMATED MOVEMENT AUDIT ---
            if len(final_images) > 1:
                # Compare the first frame to the last frame
                first_frame = final_images[0].astype(np.float32)
                last_frame = final_images[-1].astype(np.float32)
                pixel_difference = np.mean(np.abs(last_frame - first_frame))
                
                print(f"📊 DATA AUDIT: Max Pixel = {np.max(final_images)}")
                
                if pixel_difference == 0:
                    print(f"❌ CRITICAL WARNING: Episode {episode_id} is FROZEN! The images did not change.")
                else:
                    print(f"✅ MOVEMENT VERIFIED: Pixel change score = {pixel_difference:.2f}")
            # -----------------------------------
            
            with h5py.File(self.h5_path, 'a') as f:
                name = f"episode_{episode_id}"
                if name in f: del f[name]
                grp = f.create_group(name)
                grp.create_dataset("images", data=final_images, compression="gzip")
                grp.create_dataset("joints", data=np.array(obs_joints))
                print(f"💾 Successfully written {name} to Disk.")

# --- Scene Setup ---
world = World(stage_units_in_meters=1.0)

# Add Ground Plane
world.scene.add_default_ground_plane()

# Native Replicator Lighting
rep_light = rep.create.light(
    light_type="distant", 
    intensity=3000.0, 
    rotation=(315, 0, 0)
)

robot = world.scene.add(Franka(prim_path="/World/Robot", name="franka"))
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube", position=np.array([0.4, 0.0, 0.05]), size=0.05))

# Camera Setup
rp_cam = rep.create.camera(position=(1.0, 0.5, 0.8), look_at=(0.4, 0.0, 0.0))
render_product = rep.create.render_product(rp_cam, resolution=(512, 512))

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

world.reset()
harvester = ScramblerHarvester()

for i in range(5):
    harvester.collect_episode(i, world, robot, cube, rgb_annotator)
    world.reset()

print("\n🚀 DATA PIPELINE COMPLETE. Check terminal for Movement Verification.")
simulation_app.close()