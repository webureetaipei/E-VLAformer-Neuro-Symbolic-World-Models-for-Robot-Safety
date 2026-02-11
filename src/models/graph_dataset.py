import h5py
import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Callable

class EVLAGraphDataset(Dataset):
    """
    Task 11: Graph Dataset Loader for Parallel Sim-to-Real.
    Converts physical tensors into Structured Graphs (G = {V, E}).
    """
    def __init__(self, 
                 h5_path: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None):
        super(EVLAGraphDataset, self).__init__(None, transform, pre_transform)
        self.h5_path = h5_path
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found at {h5_path}")
            
        self.db = h5py.File(h5_path, 'r')
        self.keys = sorted(list(self.db.keys()))

    def len(self) -> int:
        return len(self.keys)

    def get(self, idx: int) -> Data:
        group = self.db[self.keys[idx]]
        
        # Node Features: [pos_x, pos_y, pos_z, mass_estimate, type_id]
        x = torch.tensor(group['node_features'][:], dtype=torch.float)
        
        # Edge Index: COO format connectivity
        edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)
        
        # Target Labels: Action tokens (e.g., joint deltas)
        y = torch.tensor(group['action_tokens'][:], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.sim_path = group.attrs.get('sim_path', 'unknown_sim_node')
        data.hw_id = group.attrs.get('hw_id', -1)

        return data

# ==========================================
# SMOKE TEST & DUMMY DATA GENERATOR
# ==========================================
def generate_dummy_h5(path: str):
    """Generates a fake HDF5 file to verify Task 11 logic."""
    with h5py.File(path, 'w') as f:
        for i in range(5):  # Create 5 fake episodes
            group = f.create_group(f"episode_{i}")
            # 3 nodes: Base, Arm, Object
            group.create_dataset('node_features', data=np.random.rand(3, 5)) 
            # Simple connectivity: 0-1, 1-2
            group.create_dataset('edge_index', data=np.array([[0, 1, 1, 2], [1, 0, 2, 1]])) 
            # 4 action tokens (for 4 motors)
            group.create_dataset('action_tokens', data=np.random.rand(4))
            group.attrs['sim_path'] = f"/World/Robot_{i}"
            group.attrs['hw_id'] = i

if __name__ == "__main__":
    dummy_path = "dummy_task11.h5"
    
    print("--- Task 11 Smoke Test Starting ---")
    
    # 1. Setup Dummy Data
    generate_dummy_h5(dummy_path)
    print(f"[1/3] Generated dummy data at {dummy_path}")
    
    # 2. Test Dataset Loading
    try:
        dataset = EVLAGraphDataset(h5_path=dummy_path)
        print(f"[2/3] Dataset initialized. Length: {len(dataset)}")
        
        # 3. Test Retrieval
        sample = dataset[0]
        print(f"[3/3] Successfully retrieved first graph:")
        print(f"      - Node Features Shape: {sample.x.shape}")
        print(f"      - Edge Index Shape: {sample.edge_index.shape}")
        print(f"      - Action Targets: {sample.y}")
        print(f"      - Sim Path Metadata: {sample.sim_path}")
        
        print("\n✅ Task 11 Verification PASSED.")
    except Exception as e:
        print(f"\n❌ Task 11 Verification FAILED: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path) # Cleanup