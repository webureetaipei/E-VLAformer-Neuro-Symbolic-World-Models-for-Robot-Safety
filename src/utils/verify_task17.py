# src/utils/verify_task17.py
from src.models.graph_memory import GraphMemoryBuffer
import torch
import sys

def test_object_permanence():
    print("\n--- 🧠 Task 17: Graph Memory Persistence Test ---")
    
    # Initialize buffer with a 5-frame memory
    buffer = GraphMemoryBuffer(persistence_threshold=5)
    
    # Simulate Frame 1: Robot sees the Red Cube (Node ID: 99)
    print("[Frame 1] Visual: Red Cube detected.")
    cube_feat = torch.randn(1, 32)
    buffer.update(current_nodes=[99], current_features=cube_feat)
    
    # Simulate Frames 2-5: The "Lid Test" (Cube is hidden)
    for f in range(2, 6):
        # Vision reports nothing, but the buffer should remember
        buffer.update(current_nodes=[], current_features=torch.tensor([]))
        ids, _ = buffer.get_persistent_graph()
        
        status = "RETAINED" if 99 in ids else "LOST"
        print(f"[Frame {f}] Visual: Occluded | Memory: {status} (IDs: {ids})")

    # Final Verification
    ids, _ = buffer.get_persistent_graph()
    if 99 in ids:
        print("\n✅ Task 17 SUCCESS: Object Permanence Verified.")
        print("The GNN still 'sees' the cube behind the lid.")
    else:
        print("\n❌ Task 17 FAILED: Memory leaked.")

if __name__ == "__main__":
    test_object_permanence()