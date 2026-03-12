import torch
import torch.nn as nn

class EVLAPolicyHead(nn.Module):
    """
    Task 21: VLA Policy Head (Unified Version)
    Processes the 548-dim multimodal latent into 4-DOF Joint Deltas.
    """
    def __init__(self, input_dim=548, joint_dim=4, hidden_dim=256):
        super(EVLAPolicyHead, self).__init__()
        
        # 1. Initial Projection from the 548-dim fused latent
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        
        # 2. Residual Reasoning Blocks for deep stability
        self.res_block1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_block2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Action Regression Head (Outputs delta changes for 4 joints)
        self.action_head = nn.Linear(hidden_dim, joint_dim)
        
        # 4. Normalization & Activation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, fused_latent):
        """
        fused_latent: [Batch, 548] vector from the MultimodalFusionLayer
        """
        # Initial projection
        x = self.gelu(self.fc_input(fused_latent))
        x = self.layer_norm(x)
        
        # Residual Reasoning (Skip connection)
        identity = x
        x = self.gelu(self.res_block1(x))
        x = self.res_block2(x) + identity
        x = self.layer_norm(x)
        
        # Predict Joint Deltas (Scaled to [-1, 1] for motor safety)
        actions = self.tanh(self.action_head(x))
        return actions