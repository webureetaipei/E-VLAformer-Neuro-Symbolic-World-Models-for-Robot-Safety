import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from src.models.graph_dataset import EVLAGraphDataset 
from src.models.gnn_processor import EVLAGNNProcessor 

def run_task15_visualization():
    print("🧠 Starting Task 15: Graph Latent Visualization...")
    
    # 1. Hardware & Model Setup
    device = torch.device("cpu")
    model = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32).to(device)
    model.eval()

    # 2. Load Dataset
    data_path = 'data/output/batch_v1/sim_data_batch_001.hdf5'
    dataset = EVLAGraphDataset(h5_path=data_path)
    
    num_to_process = min(len(dataset), 500)
    latents, labels = [], []

    print(f"🔄 Extracting Latents from {num_to_process} samples...")
    with torch.no_grad():
        for i in range(num_to_process):
            try:
                data = dataset[i]
                # Forward pass: Graph -> 32-dim Embedding
                out = model(data.x, data.edge_index)
                
                # Grip-centric Analysis: Focus on the gripper node (index 3)
                node_idx = 3 if out.size(0) > 3 else out.size(0) - 1
                latents.append(out[node_idx].numpy())
                labels.append(data.collision_event.item())
            except Exception as e:
                continue

    if len(np.unique(labels)) < 1:
        print("❌ Error: Not enough variety in labels to visualize clusters.")
        return

    # 3. t-SNE Manifold Learning
    print(f"📉 Reducing 32D -> 2D via t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(latents)-1), 
                init='pca', learning_rate='auto', random_state=42)
    vis_data = tsne.fit_transform(np.array(latents))

    # 4. Scientific Metrics
    if len(np.unique(labels)) > 1:
        score = silhouette_score(vis_data, labels)
        print(f"📊 Latent Separation Score (Silhouette): {score:.4f}")

    # 5. Plotting
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create the scatter plot
    # The hue='labels' will color dots by Safe (0) vs Collision (1)
    scatter = sns.scatterplot(
        x=vis_data[:,0], y=vis_data[:,1], hue=labels,
        palette='viridis', s=100, alpha=0.8, edgecolor='black'
    )
    
    plt.title("E-VLAformer: GNN Topology Analysis (Task 15)", fontsize=16)
    plt.xlabel("Latent Manifold 1")
    plt.ylabel("Latent Manifold 2")
    
    # Legend formatting
    handles, _ = scatter.get_legend_handles_labels()
    plt.legend(handles, ['Safe', 'Collision'], title="Physical State")

    # Save Output
    os.makedirs('docs/reports', exist_ok=True)
    plt.savefig('docs/reports/gnn_latent_clusters.png', dpi=300)
    print("✅ SUCCESS: Visualization saved to docs/reports/gnn_latent_clusters.png")

if __name__ == "__main__":
    run_task15_visualization()