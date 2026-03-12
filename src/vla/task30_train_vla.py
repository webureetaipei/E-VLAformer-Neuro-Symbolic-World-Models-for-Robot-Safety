import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # 👈 Added for "Smart" adjustment

from src.data.task30_evla_dataset import EVLADataset
from src.models.task13_gnn_processor import EVLAGNNProcessor
from src.models.fusion_layer import MultimodalFusionLayer
from src.models.task21_vla_policy_head import EVLAPolicyHead

def train_vla():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32 # Keep at 32 for stability
    INITIAL_LR = 1e-4
    EPOCHS = 80     # 👈 Increased epochs to let the scheduler work
    DATASET_PATH = "data/output/task30_training_master.h5"
    WEIGHT_SAVE_DIR = "models/weights/"
    
    os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
    print(f"⚙️ Initializing ADVANCED SMART Training on: {DEVICE}")

    dataset = EVLADataset(DATASET_PATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gwm_processor = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32).to(DEVICE)
    fusion_layer = MultimodalFusionLayer().to(DEVICE)
    policy_head = EVLAPolicyHead().to(DEVICE)
    
    optimizer = optim.Adam(
        list(gwm_processor.parameters()) + 
        list(fusion_layer.parameters()) + 
        list(policy_head.parameters()), 
        lr=INITIAL_LR,
        weight_decay=1e-5 # 👈 Added weight decay to prevent overfitting
    )

    # 🧠 THE SECRET SAUCE: Scheduler
    # This will cut the learning rate by 0.5 every time the loss stops dropping for 3 epochs
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Combined Loss: Huber Loss is better than MSE for robotic actions (less sensitive to outliers)
    criterion = nn.HuberLoss() 

    print("\n🚀 Commencing Advanced BC Optimization...")
    
    for epoch in range(EPOCHS):
        gwm_processor.train()
        fusion_layer.train()
        policy_head.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            vision = batch['vision'].to(DEVICE)
            lang = batch['lang'].to(DEVICE)
            proprio = batch['proprio'].to(DEVICE)
            nodes = batch['nodes'].to(DEVICE)
            edges = batch['edges'].to(DEVICE)
            target = batch['target'].to(DEVICE)

            optimizer.zero_grad()
            
            # Graph Batching
            B, num_nodes, node_feat = nodes.shape
            nodes_flat = nodes.view(-1, node_feat)
            offsets = torch.arange(B, device=DEVICE) * num_nodes
            edges_offset = edges + offsets.view(-1, 1, 1)
            edges_flat = edges_offset.permute(1, 0, 2).reshape(2, -1)
            
            gwm_node_features = gwm_processor(nodes_flat, edges_flat) 
            gwm_latent = gwm_node_features.view(B, num_nodes, -1).mean(dim=1)
            
            fused_latent = fusion_layer(vision, lang, gwm_latent, proprio)
            predicted_action = policy_head(fused_latent)
            
            loss = criterion(predicted_action, target)
            loss.backward()
            
            # Gradient Clipping: Prevents the model from "exploding" during complex perturbations
            torch.nn.utils.clip_grad_norm_(fusion_layer.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        
        # 📉 Update the scheduler with the new loss
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(WEIGHT_SAVE_DIR, f"evla_advanced_epoch{epoch+1}.pth")
            torch.save({
                'fusion_state_dict': fusion_layer.state_dict(),
                'policy_state_dict': policy_head.state_dict(),
                'gwm_state_dict': gwm_processor.state_dict(),
            }, save_path)

    print("\n🏁 Advanced Training Complete.")

if __name__ == "__main__":
    train_vla()