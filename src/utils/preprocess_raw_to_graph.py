import h5py
import numpy as np
import torch
import os

def process_raw_to_graph(file_path):
    print(f"🛠️ Preprocessing: {file_path}")
    
    with h5py.File(file_path, 'a') as f:  # 'a' means append/update
        # Check if we have the raw data
        if 'rgb' not in f:
            print("❌ Error: No 'rgb' dataset found.")
            return

        num_samples = f['rgb'].shape[0]
        
        # 1. Create Node Features (Mocking 5 features for 4 nodes)
        # In a real setup, you'd extract joint positions from metadata
        # Here we create a placeholder: [x, y, z, mass, type]
        if 'node_features' not in f:
            print(f"📝 Generating node_features for {num_samples} samples...")
            node_feats = np.random.rand(num_samples, 4, 5).astype(np.float32)
            f.create_dataset('node_features', data=node_feats)

        # 2. Create Edge Index (Standard 4-node kinematic chain)
        if 'edge_index' not in f:
            print("📝 Generating edge_index...")
            # Connections: 0-1, 1-2, 2-3 (Base to Gripper)
            edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
            # We tile this for every sample
            edge_indices = np.tile(edges, (num_samples, 1, 1))
            f.create_dataset('edge_index', data=edge_indices)

        # 3. Create Action Tokens (Dummy motor commands)
        if 'action_tokens' not in f:
            print("📝 Generating action_tokens...")
            actions = np.random.rand(num_samples, 4).astype(np.float32)
            f.create_dataset('action_tokens', data=actions)

    print("✅ Preprocessing Complete. HDF5 is now 'Graph-Ready'.")

if __name__ == "__main__":
    target_file = 'data/output/batch_v1/sim_data_batch_001.hdf5'
    if os.path.exists(target_file):
        process_raw_to_graph(target_file)
    else:
        print(f"❌ Could not find {target_file}")