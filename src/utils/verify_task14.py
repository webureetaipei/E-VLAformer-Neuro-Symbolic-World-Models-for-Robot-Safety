import torch
from src.models.gnn_processor import EVLAGNNProcessor
from src.models.fusion_layer import EVLAMultimodalFusion

def verify_full_fusion():
    print("--- Task 14: Full Multimodal Integration Test ---")
    
    # Mock inputs
    vis_tokens = torch.randn(1, 256, 512)   # Simulated ViT output
    graph_nodes = torch.randn(4, 5)         # 4 nodes from Task 11
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]) # Task 12
    
    # Models
    gnn = EVLAGNNProcessor()
    fusion = EVLAMultimodalFusion()
    
    # Step 1: GNN Processing
    graph_emb = gnn(graph_nodes, edge_index) # [4, 32]
    
    # Step 2: Fusion
    fused_out = fusion(vis_tokens, graph_emb)
    
    print(f"Fused Output Shape: {fused_out.shape}")
    if fused_out.shape == (1, 256, 512):
        print("✅ Task 14 SUCCESS: Vision and Graph Physics successfully fused!")

if __name__ == "__main__":
    verify_full_fusion()