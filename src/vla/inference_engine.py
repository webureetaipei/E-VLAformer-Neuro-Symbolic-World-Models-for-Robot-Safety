import torch
import numpy as np
from src.vla.language_handler import LanguageHandler
from src.vla.proprioception_handler import ProprioceptionHandler

class InferenceEngine:
    """
    Task 24: Inference Engine
    Synchronizes GNN, Proprioception, and Language streams to generate 
    real-time motor actions using the VLA Policy Head.
    """
    def __init__(self, policy_head_path=None):
        print("🚀 Initializing E-VLAformer Inference Engine...")
        
        # 1. Initialize Handlers
        self.lang_handler = LanguageHandler()
        self.proprio_handler = ProprioceptionHandler()
        
        # 2. Load Policy Head (Task 21)
        # In a real scenario, we load weights here. For now, we simulate the forward pass.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📟 Execution Device: {self.device}")

    def step(self, raw_joints, instruction, gnn_latent):
        """
        Performs a single multimodal inference step.
        """
        # A. Process Language (512-dim)
        lang_embed = self.lang_handler.encode(instruction).to(self.device)
        
        # B. Process Proprioception (4-dim)
        joint_tensor = self.proprio_handler.normalize(raw_joints).to(self.device)
        
        # C. Prepare GNN Latent (32-dim)
        if not isinstance(gnn_latent, torch.Tensor):
            gnn_latent = torch.tensor(gnn_latent, dtype=torch.float32).to(self.device)
        if gnn_latent.ndim == 1:
            gnn_latent = gnn_latent.unsqueeze(0)

        # D. Multimodal Fusion (Total: 548-dim)
        # [GNN (32) + Joints (4) + Lang (512)]
        fusion_vector = torch.cat([gnn_latent, joint_tensor, lang_embed], dim=-1)
        
        print(f"🔗 Fusion Vector Generated: {fusion_vector.shape}")
        
        # E. Policy Forward Pass (Simulated for Task 24 Smoke Test)
        # In Task 25, this calls the actual model.forward()
        with torch.no_grad():
            # Mocking the 4-DOF Delta output
            action_delta = torch.tanh(torch.randn(1, 4)) 
            
        return action_delta

if __name__ == "__main__":
    engine = InferenceEngine()
    
    # Mock Data for Task 24 Certification
    mock_joints = [0.0, -10.0, 45.0, 0.0]
    mock_instruction = "Pick up the red cube"
    mock_gnn = np.random.randn(32) # Simulated world state from Phase 2
    
    print("\n--- Task 24: Inference Engine Smoke Test ---")
    action = engine.step(mock_joints, mock_instruction, mock_gnn)
    
    print(f"Input Instruction: '{mock_instruction}'")
    print(f"Predicted Action Delta: {action.numpy()}")
    print("\n✅ SUCCESS: Inference Engine synchronized all 548 dimensions.")