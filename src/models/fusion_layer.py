import torch
import torch.nn as nn

class MultimodalFusionLayer(nn.Module):
    """
    Task 14: Multimodal Fusion Layer (The Universal Adapter)
    1. Converts raw RGB images into 512-dim Vision Tokens using a CNN encoder.
    2. Aligns GNN physical embeddings with Vision using Cross-Attention.
    3. Concatenates all modalities and projects them into a 548-dim vector for the Policy Head.
    """
    def __init__(self, vis_dim=512, lang_dim=512, graph_dim=32, joint_dim=4, num_heads=8):
        super(MultimodalFusionLayer, self).__init__()
        
        # 1. Lightweight Vision Encoder
        # Compresses [Batch, 3, H, W] RGB images into a [Batch, 512] feature vector
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)), # Forces any resolution down to a 1x1 spatial size
            nn.Flatten(),
            nn.Linear(64, vis_dim) # Maps the flattened features to 512 dimensions
        )
        
        # 2. Cross-Attention: Vision queries the GWM physical structure
        self.graph_projection = nn.Linear(graph_dim, vis_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(vis_dim)
        
        # 3. Final Projection
        # Calculated total concatenated dimension: 
        # Fused Vision (512) + Language (512) + Graph (32) + Joints (4) = 1060 dimensions
        total_concat_dim = vis_dim + lang_dim + graph_dim + joint_dim
        
        # Maps the 1060-dim concatenated vector down to the 548-dim latent space required by the Policy Head
        self.final_projection = nn.Linear(total_concat_dim, 548)

    def forward(self, vision_rgb, lang_embed, gnn_embed, joint_state):
        """
        Inputs:
        vision_rgb:  [Batch, 3, H, W] (or [Batch, H, W, 3] which will be automatically corrected)
        lang_embed:  [Batch, 512]
        gnn_embed:   [Batch, 32]
        joint_state: [Batch, 4]
        """
        # Step A: Handle channel dimension formatting
        # If the input tensor has the channels at the end (e.g., H, W, 3), permute to (3, H, W)
        if vision_rgb.shape[-1] == 3:
            vision_rgb = vision_rgb.permute(0, 3, 1, 2)
            
        # Step B: Convert raw images to Vision Tokens
        vis_feat = self.vision_encoder(vision_rgb)         # Outputs: [Batch, 512]
        vis_tokens = vis_feat.unsqueeze(1)                 # Adds sequence dimension: [Batch, 1, 512]
        
        # Step C: Project Graph features to match Vision dimension
        graph_feat = self.graph_projection(gnn_embed).unsqueeze(1) # Outputs: [Batch, 1, 512]
        
        # Step D: Cross-Attention
        # The Vision token acts as the Query looking for relevant physical constraints (Keys/Values) from the Graph
        attn_output, _ = self.cross_attn(query=vis_tokens, key=graph_feat, value=graph_feat)
        
        # Add residual connection, normalize, and remove the sequence dimension
        fused_vis_graph = self.norm(vis_tokens + attn_output).squeeze(1) # Outputs: [Batch, 512]
        
        # Step E: The Ultimate Concatenation
        # Combine Fused Vision-Graph (512) + Language (512) + Original Graph (32) + Proprioception (4)
        all_features = torch.cat([fused_vis_graph, lang_embed, gnn_embed, joint_state], dim=-1) # Outputs: [Batch, 1060]
        
        # Step F: Project down to the final target dimension
        final_latent = self.final_projection(all_features) # Outputs: [Batch, 548]
        
        return final_latent