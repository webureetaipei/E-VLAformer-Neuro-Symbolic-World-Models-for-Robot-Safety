import numpy as np
import sys
import os

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.franka import Franka 
from omni.isaac.franka.controllers import RMPFlowController 
from omni.isaac.core.utils.types import ArticulationAction 

class PickAndPlaceMachine:
    def __init__(self):
        self.state = "HOVER"  
        self.gripper_closed = False
        self.state_steps = 0
        # Final drop location on the ground
        self.place_pos = np.array([0.4, -0.3, 0.035]) 
        self.locked_target = None
        
    def get_next_action(self, current_pos, target_pos):
        self.state_steps += 1 
        
        if self.state == "HOVER" or self.locked_target is None:
            self.locked_target = target_pos
            
        hover_pos = self.locked_target + np.array([0, 0, 0.20]) 
        grab_pos = self.locked_target + np.array([0, 0, -0.02])  
        lift_pos = self.locked_target + np.array([0, 0, 0.35])
        
        # Placing waypoints
        move_pos = self.place_pos + np.array([0, 0, 0.35]) # High travel
        drop_pos = self.place_pos + np.array([0, 0, 0.05]) # Low placement

        dist_xy = np.linalg.norm(current_pos[:2] - self.locked_target[:2])
        goal = hover_pos

        if self.state == "HOVER":
            goal = hover_pos
            self.gripper_closed = False 
            if dist_xy < 0.02 or self.state_steps > 200: 
                print("\n🎯 ALIGNED. Dropping to grab...")
                self.state = "DESCEND_GRAB"
                self.state_steps = 0

        elif self.state == "DESCEND_GRAB":
            goal = grab_pos 
            self.gripper_closed = False 
            if current_pos[2] < 0.06 or self.state_steps > 300: 
                print("\n✊ GRABBING...")
                self.state = "GRIP"
                self.state_steps = 0
                
        elif self.state == "GRIP":
            goal = grab_pos 
            self.gripper_closed = True 
            if self.state_steps > 120: 
                print("\n🚀 LIFTING...")
                self.state = "LIFT"
                self.state_steps = 0
        
        elif self.state :
            if self.state == "LIFT":
                goal = lift_pos 
                self.gripper_closed = True 
                if current_pos[2] > (lift_pos[2] - 0.03) or self.state_steps > 200:
                    print("\n🚚 MOVING TO DROP ZONE...")
                    self.state = "MOVE"
                    self.state_steps = 0
                    
            elif self.state == "MOVE":
                goal = move_pos
                self.gripper_closed = True 
                dist_place_xy = np.linalg.norm(current_pos[:2] - self.place_pos[:2])
                if dist_place_xy < 0.04 or self.state_steps > 300:
                    print("\n⬇️ LOWERING BALL TO GROUND...")
                    self.state = "DESCEND_PLACE"
                    self.state_steps = 0

            elif self.state == "DESCEND_PLACE":
                goal = drop_pos
                self.gripper_closed = True 
                # Wait until ball is close to floor
                if current_pos[2] < 0.12 or self.state_steps > 250:
                    print("\n👐 RELEASING BALL ON GROUND.")
                    self.state = "RELEASE"
                    self.state_steps = 0
                    
            elif self.state == "RELEASE":
                goal = drop_pos
                self.gripper_closed = False 
                if self.state_steps > 100:
                    print("\n✅ TASK COMPLETE.")
                    self.state = "DONE"
                    self.state_steps = 0
                    
            elif self.state == "DONE":
                # Retreat to a safe height
                goal = move_pos
                self.gripper_closed = False 

        return goal, self.gripper_closed

# --- Scene Setup ---
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

robot = world.scene.add(Franka(prim_path="/World/Robot", name="franka"))

ball = world.scene.add(DynamicSphere(
    prim_path="/World/Ball", 
    name="ball", 
    position=np.array([0.4, 0.1, 0.035]), 
    radius=0.035, 
    color=np.array([1, 0, 0])
))

world.reset()
ball.set_mass(0.01) 

controller = RMPFlowController(name="rmp", robot_articulation=robot)
machine = PickAndPlaceMachine()

print("\n🚀 Starting Task 28: Complete Ground-to-Ground Sequence")
print("-" * 50)

for i in range(4000): # Increased loop for the extra descend step
    world.step(render=True)
    if world.is_stopped(): break

    ee_pos = robot.end_effector.get_world_pose()[0]
    ball_pos = ball.get_world_pose()[0]
    
    if i % 100 == 0:
        print(f"Frame {i:04d} | State: {machine.state:<13} | Z Height: {ee_pos[2]:.3f}m")
    
    goal, should_close = machine.get_next_action(ee_pos, ball_pos)
    
    actions = controller.forward(target_end_effector_position=goal)
    if actions is not None:
        robot.apply_action(actions)
        
    if should_close:
        gripper_action = ArticulationAction(joint_positions=np.array([0.0, 0.0]))
    else:
        gripper_action = ArticulationAction(joint_positions=np.array([0.04, 0.04]))
        
    robot.gripper.apply_action(gripper_action)

print("\n✅ Simulation loop ended.")
simulation_app.close()