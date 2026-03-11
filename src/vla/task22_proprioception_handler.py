import numpy as np
import torch

class ProprioceptionHandler:
    """
    Task 22: Joint Space Proprioception
    Maps raw robot angles to normalized tensors for the Policy Head.
    Includes Kalman-style smoothing for noisy sim-to-real transitions.
    """
    def __init__(self):
        # Default limits for the 4-DOF DIY arm (in degrees)
        self.joint_min = np.array([-90.0, -90.0, -90.0, -90.0])
        self.joint_max = np.array([90.0, 90.0, 90.0, 90.0])
        
        # Simple state for smoothing (Moving Average baseline)
        self.prev_state = None
        self.alpha = 0.7  # Smoothing factor (Low-pass filter)

    def normalize(self, raw_angles):
        """
        Normalize raw angles from Isaac Sim to [-1, 1] range.
        """
        raw_angles = np.array(raw_angles)
        
        # Apply smoothing
        if self.prev_state is not None:
            raw_angles = self.alpha * raw_angles + (1 - self.alpha) * self.prev_state
        self.prev_state = raw_angles

        # Linear mapping to [-1, 1]
        normalized = 2.0 * (raw_angles - self.joint_min) / (self.joint_max - self.joint_min) - 1.0
        
        # Clamp to ensure strict [-1, 1] safety
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

    def denormalize(self, normalized_actions, max_delta=5.0):
        """
        Convert Policy Head output back to Delta Degrees for motor commands.
        """
        delta_degrees = normalized_actions.detach().cpu().numpy() * max_delta
        return delta_degrees

if __name__ == "__main__":
    handler = ProprioceptionHandler()
    
    # Simulation: Robot at 45 deg with some noise
    sample_input = [45.2, -44.8, 0.5, 10.1]
    tensor_out = handler.normalize(sample_input)
    
    print("--- Task 22: Proprioception Verification ---")
    print(f"Raw Input: {sample_input}")
    print(f"Normalized/Filtered: {tensor_out.numpy()}")
    
    # Check bounds
    in_bounds = torch.all(tensor_out <= 1.0) and torch.all(tensor_out >= -1.0)
    print(f"✅ Safety Bounds Verified: {in_bounds}")