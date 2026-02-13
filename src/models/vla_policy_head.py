import torch
import torch.nn as nn
import torch.nn.functional as F

class EVLAPolicyHead(nn.Module):
    """
    Task 21: VLA Policy Head
    Fuses Multimodal Latents -> Joint Delta Actions
    """
    def __init__(self, latent_dim=32, joint_dim=4, lang_dim=512, hidden_dim=256):
        super(EVLAPolicyHead, self).__init__()
        
        # 1. Multimodal Fusion Layer
        # Inputs: GNN Latent (32) + Proprioception (4) + Language (512)
        input_dim = latent_dim + joint_dim + lang_dim
        
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        
        # 2. Residual Reasoning Blocks
        self.res_block1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_block2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Action Regression Head
        # Predicts delta change for 4-DOF joints
        self.action_head = nn.Linear(hidden_dim, joint_dim)
        
        # 4. Normalization & Activation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, gnn_embed, joint_state, lang_embed):
        # Concatenate all sensory streams
        # Shape: [Batch, latent + joint + lang]
        x = torch.cat([gnn_embed, joint_state, lang_embed], dim=-1)
        
        # Initial projection
        x = self.gelu(self.fc_input(x))
        x = self.layer_norm(x)
        
        # Residual Reasoning (Skip connections for deep stability)
        identity = x
        x = self.gelu(self.res_block1(x))
        x = self.res_block2(x) + identity
        x = self.layer_norm(x)
        
        # Predict Joint Deltas (Scaled to [-1, 1] for motor safety)
        actions = self.tanh(self.action_head(x))
        return actions

if __name__ == "__main__":
    # Integration Smoke Test
    model = EVLAPolicyHead()
    
    # Mock Tensors: [Batch Size, Dimension]
    mock_gnn = torch.randn(1, 32)
    mock_joints = torch.randn(1, 4)
    mock_lang = torch.randn(1, 512)
    
    out = model(mock_gnn, mock_joints, mock_lang)
    
    print("--- Task 21: Policy Head Initialization ---")
    print(f"Action Vector: {out.detach().numpy()}")
    print(f"✅ SUCCESS: Predicted {out.shape[1]} joint deltas.")