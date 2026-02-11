import torch
from src.models.graph_dataset import EVLAGraphDataset
from src.models.graph_builder import RelationalGraphBuilder
from torch_geometric.data import Data

def run_integration_test():
    builder = RelationalGraphBuilder(contact_threshold=0.1) # 10cm threshold
    
    # 1. Create Mock Data (Simulating a Gripper near a Cube)
    # Nodes: [Base, Joint, Gripper, Cube]
    # Features: [x, y, z, mass, type_id]
    nodes = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0], # Base
        [0.0, 0.1, 0.2, 0.5, 1], # Joint
        [0.0, 0.2, 0.4, 0.2, 2], # Gripper (at z=0.4)
        [0.0, 0.2, 0.45, 0.1, 3] # Cube (at z=0.45 - Close to gripper!)
    ], dtype=torch.float)
    
    data = Data(x=nodes)
    
    # 2. Run the Builder
    print("--- Phase 2: Relational Graph Construction ---")
    updated_data = builder.update_graph(data)
    
    # 3. Verify Edges
    print(f"Total Edges Found: {updated_data.edge_index.shape[1]}")
    # Kinematic edges (6) + Contact edges (2) = 8
    if updated_data.edge_index.shape[1] >= 8:
        print("✅ Task 12 SUCCESS: Kinematic and Contact edges generated.")
    else:
        print("❌ Task 12 FAILED: Missing edges.")

if __name__ == "__main__":
    run_integration_test()