import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool  # <--- FIX: Direct import
from src.models.graph_dataset import EVLAGraphDataset
from src.models.gnn_processor import EVLAGNNProcessor
import os

# --- Hyperparameters for Identity Preservation ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 
EPOCHS = 50            
TEMPERATURE = 0.1      # Increased to 0.1 to encourage cluster overlap (Stability)

def contrastive_loss(out, labels, temperature=0.1):
    """
    Supervised Contrastive Loss tuned for Identity Mapping.
    """
    out = F.normalize(out, dim=1)
    similarity_matrix = torch.matmul(out, out.t()) / temperature
    
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()
    
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, 
        torch.arange(mask.shape[0]).view(-1, 1).to(mask.device), 0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {device} | Implementing Identity Collapse (Task 19.1)...")

    h5_path = 'data/raw/task18_occlusion_test_001.h5'
    if not os.path.exists(h5_path):
        print(f"❌ Error: {h5_path} not found.")
        return

    dataset = EVLAGraphDataset(h5_path=h5_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print(f"--- 🧠 Goal: Merge 'Visible' and 'Hidden' into a single Latent Identity ---")
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # 1. Feature Perturbation (Augmentation)
            # Adds small noise to help the model ignore 'sensory flicker'
            noise = torch.randn_like(data.x) * 0.01
            x_augmented = data.x + noise
            
            out = model(x_augmented, data.edge_index)
            z = global_mean_pool(out, data.batch)
            # 2. CRITICAL CHANGE: Identity Mapping
            # Instead of training against data.occluded_flag (which creates 2 clusters),
            # we train against a constant ID. This forces 'Visible' and 'Hidden' 
            # to occupy the exact same coordinate in the latent manifold.
            identity_labels = torch.zeros(z.size(0), device=device) 
            
            loss = contrastive_loss(z, identity_labels, temperature=TEMPERATURE)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Identity Loss: {total_loss/len(loader):.4f}")

    os.makedirs('models/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/weights/gnn_contrastive_beta.pth')
    print("✅ Identity Mapping Complete! Weights saved.")

if __name__ == "__main__":
    train()