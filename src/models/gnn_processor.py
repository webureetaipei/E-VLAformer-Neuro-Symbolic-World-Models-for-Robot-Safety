import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class EVLAGNNProcessor(torch.nn.Module):
    """
    Task 13: GNN Message Passing Layers.
    Processes the Relational Graph to predict physical state transitions.
    """
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=32):
        super(EVLAGNNProcessor, self).__init__()
        
        # Layer 1: Aggregates features from immediate neighbors (e.g., Joint to Link)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        
        # Layer 2: Aggregates multi-hop information (e.g., Base to Gripper)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Layer 3: Final physical reasoning layer
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        # 1. First Message Passing block with non-linear activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 2. Second Message Passing block
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 3. Final embedding generation per node
        x = self.conv3(x, edge_index)
        
        # Optional: Global pooling if we need a single "Scene Sentiment" vector
        # scene_embedding = global_mean_pool(x, batch) 
        
        return x

if __name__ == "__main__":
    model = EVLAGNNProcessor()
    print("Task 13: EVLAGNNProcessor (GraphSAGE) initialized.")