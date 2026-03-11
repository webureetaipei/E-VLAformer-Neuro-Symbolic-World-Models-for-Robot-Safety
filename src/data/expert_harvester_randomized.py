import os
import h5py
import numpy as np
import time
import traceback
import random

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from isaacsim import SimulationApp
# Set headless=True for faster collection once you verify the visuals
simulation_app = SimulationApp({"headless": False})

# 🔥 FIX: Added UsdGeom and Gf to the imports to handle USD data types properly
from pxr import UsdLux, UsdPhysics, UsdShade, UsdGeom, Gf 
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction 
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController 

class PickAndPlaceMachine:
    def __init__(self):
        self.state = "HOVER"  
        self.gripper_closed = False
        self.state_steps = 0
        
    def get_next_action(self, current_pos, target_ball_pos):
        self.state_steps += 1 
        
        # Target Waypoints (Dynamically updated every frame)
        hover_pos = target_ball_pos + np.array([0, 0, 0.18]) 
        # Deep Grasp: ensures fingers wrap around the ball's center
        grab_pos = target_ball_pos + np.array([0, 0, 0.022]) 
        lift_pos = target_ball_pos + np.array([0, 0, 0.32])

        dist_xy = np.linalg.norm(current_pos[:2] - target_ball_pos[:2])
        goal = hover_pos

        if self.state == "HOVER":
            goal = hover_pos
            self.gripper_closed = False 
            if dist_xy < 0.015 or self.state_steps > 150: 
                self.state = "DESCEND_GRAB"
                self.state_steps = 0

        elif self.state == "DESCEND_GRAB":
            goal = grab_pos 
            self.gripper_closed = False 
            if current_pos[2] < (grab_pos[2] + 0.01) or self.state_steps > 180: 
                self.state = "GRIP"
                self.state_steps = 0
                
        elif self.state == "GRIP":
            goal = grab_pos 
            self.gripper_closed = True 
            if self.state_steps > 90: # Extra time to secure the Iron Grip
                self.state = "LIFT"
                self.state_steps = 0
        
        elif self.state == "LIFT":
            goal = lift_pos 
            self.gripper_closed = True 
            if current_pos[2] > (lift_pos[2] - 0.03) or self.state_steps > 200:
                self.state = "DONE"
                self.state_steps = 0
                
        elif self.state == "DONE":
            goal = lift_pos
            self.gripper_closed = True 

        return goal, self.gripper_closed

class ExpertHarvester:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        
        # Initial Randomized Spawn
        rand_x = random.uniform(0.38, 0.45)
        rand_y = random.uniform(-0.10, 0.10)
        self.target_pos = np.array([rand_x, rand_y, 0.035], dtype=np.float32)
        
        # Choose Scenario
        self.episode_type = random.choice(["NORMAL", "OCCLUSION", "PERTURBATION"])  
        
        self.base_data_path = r"C:\Users\Michael\evlaformer_lab\data"
        os.makedirs(self.base_data_path, exist_ok=True)

        filename = os.path.join(
            self.base_data_path,
            f"task29_{self.episode_type}_{int(time.time())}.h5"
        )

        print(f"\n🚀 [DATA ENGINE] Mode: {self.episode_type}")
        print(f"📂 [DATA ENGINE] Output: {filename}")

        self.h5_file = h5py.File(filename, "w")
        self.data_group = self.h5_file.create_group("data")
        self.step_idx = 0

        self.setup_scene()
        self.setup_camera()

        self.world.reset()
        self.robot.initialize()

        self.controller = RMPFlowController(name="rmp", robot_articulation=self.robot)
        self.machine = PickAndPlaceMachine()

        self.event_triggered = False
        self.perturbation_triggered = False
        self.counter = 0

    def setup_scene(self):
        self.world.scene.add_default_ground_plane()
        stage = self.world.stage
        
        # 🔥 THE FIX: Using UsdGeom and Gf.Vec3d to correctly set the position
        light = UsdLux.SphereLight.Define(stage, "/World/WorkspaceLight")
        light.CreateIntensityAttr(50000.0)
        light.CreateRadiusAttr(0.5)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 1.5))

        # Friction Material
        material_path = "/World/StickyMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        phys_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
        phys_api.CreateStaticFrictionAttr(2.0)
        phys_api.CreateDynamicFrictionAttr(2.0)

        # RED BALL
        self.target_ball = self.world.scene.add(
            DynamicSphere(prim_path="/World/TargetBall", name="target_ball",
                          position=self.target_pos, radius=0.035, 
                          color=np.array([1.0, 0.0, 0.0]), mass=0.01)
        )
        ball_prim = stage.GetPrimAtPath("/World/TargetBall")
        UsdPhysics.CollisionAPI.Apply(ball_prim)
        UsdShade.MaterialBindingAPI.Apply(ball_prim).Bind(material, "physics")

        # Visual-Friendly Occluder (Wall)
        prim_utils.create_prim("/World/Occluder", "Cube", 
                               position=np.array([0,0,-5]), 
                               scale=np.array([0.01, 0.5, 0.5]))
        self.occluder = self.world.scene.add(XFormPrim("/World/Occluder", name="occluder"))

        self.robot = self.world.scene.add(Franka(prim_path="/World/Robot", name="active_arm"))

    def setup_camera(self):
        # WIDE ANGLE CAMERA POSITION
        self.camera = Camera(
            prim_path="/World/Camera", 
            position=np.array([1.2, 0.0, 0.6]), 
            orientation=np.array([0.0, -0.25, 0.0, 0.95]), 
            resolution=(256, 256)
        )
        self.camera.initialize()
        
        # Configure raw USD camera attributes safely with Gf types
        usd_cam = self.camera.prim
        usd_cam.GetAttribute("focalLength").Set(15.0) 
        usd_cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10.0)) 

    def record_step(self, joint_positions):
        rgba = self.camera.get_rgba()
        if rgba is None: return
        rgb = rgba[:, :, :3].astype(np.uint8)
        step = self.data_group.create_group(f"step_{self.step_idx}")
        step.create_dataset("obs/joint_positions", data=self.robot.get_joint_positions())
        step.create_dataset("obs/image", data=rgb)
        step.create_dataset("action", data=joint_positions)
        self.step_idx += 1

    def run(self):
        try:
            # Short warmup
            for _ in range(30): self.world.step(render=True)
                
            while simulation_app.is_running():
                self.world.step(render=True)
                if not self.world.is_playing(): continue

                ee_pos = self.robot.end_effector.get_world_pose()[0]
                ball_pos, _ = self.target_ball.get_world_pose()

                # --- PERTURBATION: Ball shifts VERY early (Frame 5) ---
                if self.episode_type == "PERTURBATION":
                    if self.counter == 5 and not self.perturbation_triggered:
                        # Move the ball significantly before robot moves
                        shift = np.array([random.uniform(0.08, 0.12), random.uniform(-0.15, 0.15), 0.0])
                        new_pos = ball_pos + shift
                        self.target_ball.set_world_pose(position=new_pos)
                        print(f"!!! PERTURBATION: Ball moved to {new_pos} at START.")
                        self.perturbation_triggered = True
                        ball_pos = new_pos # Sync position for this frame

                # --- OCCLUSION: View Blocked at Frame 60 ---
                elif self.episode_type == "OCCLUSION":
                    if self.counter == 60 and not self.event_triggered:
                        self.occluder.set_world_pose(position=np.array([1.10, 0.0, 0.5]))
                        print("!!! OCCLUSION: View Blocked.")
                        self.event_triggered = True
                    if self.event_triggered:
                        self.occluder.set_world_pose(position=np.array([1.10, 0.0, 0.5]))

                # 1. Expert Logic (Dynamic Goal)
                goal, should_close = self.machine.get_next_action(ee_pos, ball_pos)
                
                # 2. Motion Generation
                actions = self.controller.forward(target_end_effector_position=goal)
                if actions: self.robot.apply_action(actions)
                
                # 3. IRON GRIP - Constant inward pressure
                if should_close:
                    grip_cmd = np.array([-0.015, -0.015]) # Stronger inward torque
                else:
                    grip_cmd = np.array([0.04, 0.04]) 
                self.robot.gripper.apply_action(ArticulationAction(joint_positions=grip_cmd))

                if self.counter % 10 == 0:
                    self.record_step(self.robot.get_joint_positions())

                self.counter += 1
                if self.machine.state == "DONE":
                    final_pos, _ = self.target_ball.get_world_pose()
                    if final_pos[2] > 0.15: 
                        print(f"\n✅ SUCCESS: {self.episode_type} Complete!")
                    else:
                        print(f"\n❌ FAILED: Ball dropped.")
                    break
                if self.counter > 950: break

        except Exception:
            print(f"\n🚨 ERROR: {traceback.format_exc()}")
        finally:
            self.h5_file.close()
            simulation_app.close()

if __name__ == "__main__":
    harvester = ExpertHarvester()
    harvester.run()