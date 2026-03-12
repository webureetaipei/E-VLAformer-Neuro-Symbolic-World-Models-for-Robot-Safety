import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class EVLADataset(Dataset):
    """
    Dataset loader for the E-VLAformer.
    Pulls real Vision, Proprioception, GWM Nodes, and Language Embeddings.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        with h5py.File(self.file_path, 'r') as f:
            # Get all step keys and sort them numerically
            self.step_keys = [k for k in f['data'].keys() if k.startswith('step_')]
            self.step_keys.sort(key=lambda x: int(x.split('_')[1]))
            self.total_frames = len(self.step_keys)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        step_name = self.step_keys[idx]

        with h5py.File(self.file_path, 'r') as f:
            group = f['data'][step_name]
            
            # --- 1. Vision & Proprioception ---
            vision_np = group['obs']['image'][()]
            # Normalize and permute to [C, H, W] for CNN if needed, 
            # though usually [H, W, C] is fine for initial fusion layers.
            vision = torch.from_numpy(vision_np).float() / 255.0
            
            proprio_np = group['obs']['joint_positions'][()]
            proprio = torch.from_numpy(proprio_np[:4]).float()
            
            # --- 2. Action (Ground Truth) ---
            target_np = group['action'][()]
            target_action = torch.from_numpy(target_np[:4]).float()
            
            # --- 3. REAL GWM & Language Data (Unmasked) ---
            # These are the signals that solve Occlusion and Perturbation
            lang = torch.from_numpy(group['lang_embed'][()]).float()
            nodes = torch.from_numpy(group['nodes'][()]).float()
            edges = torch.from_numpy(group['edges'][()]).long()

        return {
            "vision": vision,
            "lang": lang,
            "proprio": proprio,
            "nodes": nodes,
            "edges": edges,
            "target": target_action
        }