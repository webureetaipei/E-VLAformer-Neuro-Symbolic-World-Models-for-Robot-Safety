import torch
from src.models.graph_builder import RelationalGraphBuilder
from src.models.gnn_processor import EVLAGNNProcessor
from torch_geometric.data import Data

def verify_gnn_flow():
    print("--- Task 13: GNN Forward Pass Verification ---")
    
    # 1. Initialize Task 12 Builder and Task 13 Processor
    builder = RelationalGraphBuilder(contact_threshold=0.1)
    processor = EVLAGNNProcessor(in_channels=5, hidden_channels=64, out_channels=32)
    processor.eval() # Set to evaluation mode (disables dropout)

    # 2. Create Mock Input (4 nodes: Base, Joint, Gripper, Cube)
    # Features: [x, y, z, mass, type_id]
    x = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0], 
        [0.0, 0.1, 0.2, 0.5, 1], 
        [0.0, 0.2, 0.4, 0.2, 2], 
        [0.0, 0.2, 0.45, 0.1, 3] 
    ], dtype=torch.float)
    
    data = Data(x=x)

    # 3. Build the Relational Graph (Task 12 logic)
    data = builder.update_graph(data)
    print(f"Graph initialized with {data.num_edges} edges.")

    # 4. Run the GNN Processor (Task 13 logic)
    with torch.no_grad():
        output_embeddings = processor(data.x, data.edge_index)

    # 5. Validation Logic
    print(f"Output Embedding Shape: {output_embeddings.shape}")
    
    # Check if we have an embedding for every node
    if output_embeddings.shape == (4, 32):
        print("✅ Task 13 SUCCESS: GNN successfully processed the relational graph.")
    else:
        print("❌ Task 13 FAILED: Unexpected output shape.")

if __name__ == "__main__":
    verify_gnn_flow()