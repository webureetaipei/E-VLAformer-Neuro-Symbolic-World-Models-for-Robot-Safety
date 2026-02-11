import torch
import torch.nn as nn

class EVLAMultimodalFusion(nn.Module):
    """
    Task 14: Multimodal Fusion Layer.
    Aligns GNN physical embeddings with Vision-Language tokens using Cross-Attention.
    """
    def __init__(self, vis_dim=512, graph_dim=32, num_heads=8):
        super(EVLAMultimodalFusion, self).__init__()
        
        # Project Graph embeddings to match Vision token dimension
        self.graph_projection = nn.Linear(graph_dim, vis_dim)
        
        # Cross-Attention: Vision queries the Graph
        self.cross_attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads, batch_first=True)
        
        # Layer Norm and Feed-Forward for stability
        self.norm = nn.LayerNorm(vis_dim)
        self.ff = nn.Sequential(
            nn.Linear(vis_dim, vis_dim * 2),
            nn.ReLU(),
            nn.Linear(vis_dim * 2, vis_dim)
        )

    def forward(self, vis_tokens, graph_embeddings):
        """
        vis_tokens: [Batch, Sequence_Len, 512]
        graph_embeddings: [Nodes, 32] -> Needs to be reshaped to [Batch, Nodes, 32]
        """
        # 1. Project Graph features to the fusion space
        graph_feat = self.graph_projection(graph_embeddings).unsqueeze(0) # [1, Nodes, 512]
        
        # 2. Cross-Attention: (Query, Key, Value)
        # Vision tokens look at Graph features to find relevant physical constraints
        attn_output, _ = self.cross_attn(query=vis_tokens, key=graph_feat, value=graph_feat)
        
        # 3. Residual connection and Norm
        out = self.norm(vis_tokens + attn_output)
        
        # 4. Final Feed-Forward refinement
        out = out + self.ff(out)
        
        return out

if __name__ == "__main__":
    fusion = EVLAMultimodalFusion()
    print("Task 14: Multimodal Fusion Layer (Cross-Attention) initialized.")