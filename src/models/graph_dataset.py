import h5py
import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset

class EVLAGraphDataset(Dataset):
    """
    Task 11: High-Performance Adaptive Graph Loader.
    Supports both nested group structures and flat dataset arrays.
    """
    def __init__(self, h5_path: str, transform=None, pre_transform=None):
        super(EVLAGraphDataset, self).__init__(None, transform, pre_transform)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        self.db = h5py.File(h5_path, 'r')
        
        # Determine if file is Flat or Group-based
        # If 'node_features' is a root-level dataset, it's Flat.
        if 'node_features' in self.db and isinstance(self.db['node_features'], h5py.Dataset):
            self.mode = "flat"
            self.num_samples = self.db['node_features'].shape[0]
        else:
            self.mode = "group"
            self.keys = sorted(list(self.db.keys()), 
                               key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            self.num_samples = len(self.keys)

    def len(self) -> int:
        return self.num_samples

    def get(self, idx: int) -> Data:
        if self.mode == "flat":
            # Direct slicing from root-level arrays
            x_raw = self.db['node_features'][idx]
            edge_raw = self.db['edge_index'][idx]
            y_raw = self.db['action_tokens'][idx]
            coll_raw = self.db['collision_event'][idx]
        else:
            # Slicing from individual frame groups
            group = self.db[self.keys[idx]]
            x_raw = group['node_features'][()]
            edge_raw = group['edge_index'][()]
            y_raw = group['action_tokens'][()]
            coll_raw = group.attrs.get('collision_event', 0)

        # 1. Convert to Torch Tensors
        x = torch.from_numpy(np.array(x_raw)).to(torch.float)
        edge_index = torch.from_numpy(np.array(edge_raw)).to(torch.long)
        y = torch.from_numpy(np.array(y_raw)).to(torch.float)

        # 2. Geometry Correction (Ensure [2, E] shape)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        # 3. Label Standardization
        coll_val = coll_raw[0] if hasattr(coll_raw, '__len__') else coll_raw
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.collision_event = torch.tensor([coll_val], dtype=torch.long)
        return data

if __name__ == "__main__":
    print("🧪 EVLAGraphDataset: Initialized in Diagnostic Mode.")