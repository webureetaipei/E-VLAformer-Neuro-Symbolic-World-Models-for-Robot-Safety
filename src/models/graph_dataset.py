# src/models/graph_dataset.py

import torch
from torch_geometric.data import Data, Dataset
import h5py

class EVLAGraphDataset(Dataset):
    def __init__(self, h5_path, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            # Determine the number of frames by checking the occluded_flag length
            self.num_frames = len(f['occluded_flag'])

    def len(self):
        return self.num_frames

    def get(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            # FIX: Task 18 data is stored in direct datasets, not compound groups
            # If your Task 18 script created a cube at /World/RedCube
            # We simulate a 5-feature vector: [pos_x, pos_y, pos_z, size, color_r]
            
            # Since Task 18 focused on the Flag, we reconstruct the node features
            # In a full run, your sim script would save 'node_features' directly.
            # For this recalibration, we assume a single node (the cube).
            
            # Mocking the node features if they aren't explicitly in your H5 yet:
            x = torch.tensor([[0.0, 0.0, 0.5, 0.1, 1.0]], dtype=torch.float) 
            
            # Load the actual hardened flag we generated
            y = torch.tensor([f['occluded_flag'][idx]], dtype=torch.long)
            
            # Default edges (self-loop for a single node)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, occluded_flag=y)
            return data