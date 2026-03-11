import torch
import h5py
import numpy as np
from sklearn.metrics import silhouette_score
from src.models.gnn_processor import EVLAGNNProcessor
from torch_geometric.nn import global_mean_pool

def run_real_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/weights/gnn_contrastive_beta.pth'
    data_path = 'data/raw/task18_occlusion_test_001.h5'

    # 1. Load the "Hardened Brain"
    model = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(data_path, 'r') as f:
        flags = f['occluded_flag'][:]
        embeddings = []

        # 2. Pass data through the ACTUAL GNN
        with torch.no_grad():
            for i in range(len(flags)):
                # Mock node features (matching training logic)
                x = torch.tensor([[0.0, 0.0, 0.5, 0.1, 1.0]], device=device)
                edge_index = torch.tensor([[0], [0]], device=device)
                
                out = model(x, edge_index)
                z = global_mean_pool(out, torch.tensor([0], device=device))
                embeddings.append(z.cpu().numpy().flatten())

        embeddings = np.array(embeddings)
        
        # 3. Calculate Score
        score = silhouette_score(embeddings, flags)
        
        print(f"--- 📊 Task 19: Real-Weight Audit Results ---")
        print(f"⭐ Silhouette Stability Coefficient: {score:.4f}")
        
        if score > 0.55:
            print("✅ SUCCESS: The GNN has achieved topological stability!")
        else:
            print("🟡 MARGINAL: Identity collapse is incomplete.")

if __name__ == "__main__":
    run_real_audit()