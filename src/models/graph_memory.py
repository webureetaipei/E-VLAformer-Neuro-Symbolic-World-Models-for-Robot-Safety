import torch
import numpy as np

class GraphMemoryBuffer:
    def __init__(self, max_nodes=50, persistence_threshold=100):
        """
        max_nodes: Maximum objects to track
        persistence_threshold: Frames to remember an object after it disappears
        """
        self.memory = {} # {node_id: {'features': tensor, 'ttl': int}}
        self.threshold = persistence_threshold

    def update(self, current_nodes, current_features):
        """
        Updates memory with new sightings and decays old ones.
        """
        # 1. Update/Add current sightings
        for i, node_id in enumerate(current_nodes):
            self.memory[node_id] = {
                'features': current_features[i],
                'ttl': self.threshold
            }

        # 2. Decay TTL for missing nodes
        expired_nodes = []
        for node_id in self.memory:
            if node_id not in current_nodes:
                self.memory[node_id]['ttl'] -= 1
                if self.memory[node_id]['ttl'] <= 0:
                    expired_nodes.append(node_id)

        # 3. Clean up
        for node_id in expired_nodes:
            del self.memory[node_id]

    def get_persistent_graph(self):
        """
        Returns all nodes currently in memory to reconstruct the 'Mental Graph'.
        """
        ids = list(self.memory.keys())
        features = torch.stack([m['features'] for m in self.memory.values()])
        return ids, features