import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class LanguageHandler:
    """
    Task 23: Language Grounding
    Converts natural language instructions into 512-dim embeddings.
    """
    def __init__(self, model_name='all-distilroberta-v1'):
        print(f"🧠 Loading Language Encoder: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # New: Projection Layer to align with Task 21 (768 -> 512)
        self.projection = nn.Linear(768, 512)
        print("📏 Projection Layer Initialized: 768 -> 512")

    def encode(self, text_instruction):
        """
        Processes text and returns a projected 512-dim tensor.
        """
        if not text_instruction:
            text_instruction = "Stay idle"
            
        with torch.no_grad():
            raw_embedding = self.model.encode(text_instruction, convert_to_tensor=True)
            
            # Apply the projection
            if raw_embedding.ndim == 1:
                raw_embedding = raw_embedding.unsqueeze(0)
            
            # We use no_grad for inference, but the weights stay consistent
            projected_embedding = self.projection(raw_embedding)
            
        return projected_embedding

if __name__ == "__main__":
    handler = LanguageHandler()
    
    cmd = "Pick up the red cube"
    embedding = handler.encode(cmd)
    
    print("\n--- Task 23: Aligned Language Verification ---")
    print(f"Command: '{cmd}'")
    print(f"Final Embedding Shape: {list(embedding.shape)}")
    
    # Audit Check
    success = (embedding.shape[1] == 512)
    print(f"✅ DIMENSION ALIGNMENT SUCCESS: {success}")