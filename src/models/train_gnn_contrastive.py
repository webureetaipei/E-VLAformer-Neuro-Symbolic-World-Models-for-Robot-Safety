import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.models.graph_dataset import EVLAGraphDataset
from src.models.gnn_processor import EVLAGNNProcessor
import os

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TEMPERATURE = 0.07  # Controls the sharpness of the latent distribution

def contrastive_loss(out, labels, temperature=0.07):
    """
    Implements Supervised Contrastive Loss (Variation of NT-Xent).
    Encourages embeddings of the same class to cluster together.
    """
    # Normalize vectors to the unit hypersphere
    out = F.normalize(out, dim=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.matmul(out, out.t()) / temperature
    
    # Create mask: Identify samples with the same label (Positive Pairs)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()
    
    # Exclude self-contrast (diagonal elements)
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, 
        torch.arange(mask.shape[0]).view(-1, 1).to(mask.device), 0
    )
    mask = mask * logits_mask

    # Compute Log-Softmax
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    
    # Calculate Mean Log-Likelihood of positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {device} | Starting Task 16 Training Pipeline...")

    # 1. Load Dataset
    dataset = EVLAGraphDataset(h5_path='data/output/batch_v1/sim_data_batch_001.hdf5')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model & Optimizer
    model = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward: Get node embeddings
            out = model(data.x, data.edge_index)
            
            # Global Mean Pooling to obtain a single vector per graph
            from torch_geometric.nn import global_mean_pool
            z = global_mean_pool(out, data.batch)
            
            # Calculate Contrastive Loss
            loss = contrastive_loss(z, data.collision_event)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Average Loss: {total_loss/len(loader):.4f}")

    # 3. Save the trained weight "Brain"
    os.makedirs('models/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/weights/gnn_contrastive_beta.pth')
    print("✅ Training Complete! Weights saved to: models/weights/gnn_contrastive_beta.pth")

if __name__ == "__main__":
    train()