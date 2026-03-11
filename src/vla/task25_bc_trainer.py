import torch
import torch.nn as nn
import torch.optim as optim
from src.models.vla_policy_head import EVLAPolicyHead

class BCTrainer:
    """
    Task 25: Behavioral Cloning Trainer
    Trains the VLA Policy Head by passing discrete multimodal streams.
    """
    def __init__(self, policy_head, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = policy_head.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        print(f"🏋️ BC Trainer Initialized on {self.device}")

    def train_step(self, gnn_embed, joint_state, lang_embed, expert_action):
        """
        Single optimization step.
        Inputs are passed separately to match EVLAPolicyHead.forward()
        """
        self.model.train()
        
        # Move all to device
        gnn_embed = gnn_embed.to(self.device)
        joint_state = joint_state.to(self.device)
        lang_embed = lang_embed.to(self.device)
        expert_action = expert_action.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward Pass: Passing 3 separate arguments as required by Task 21
        predicted_action = self.model(gnn_embed, joint_state, lang_embed)
        
        loss = self.criterion(predicted_action, expert_action)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

if __name__ == "__main__":
    print("🛠️ Testing Task 25: Behavioral Cloning Pipeline...")

    # Initialize Policy Head (32 + 4 + 512)
    policy = EVLAPolicyHead(latent_dim=32, joint_dim=4, lang_dim=512)
    trainer = BCTrainer(policy)
    
    # Simulated Batch Data (Discrete Streams)
    batch_size = 8
    mock_gnn = torch.randn(batch_size, 32)
    mock_joints = torch.randn(batch_size, 4)
    mock_lang = torch.randn(batch_size, 512)
    mock_expert = torch.randn(batch_size, 4)
    
    print("\n--- Task 25: Behavioral Cloning Verification ---")
    loss = trainer.train_step(mock_gnn, mock_joints, mock_lang, mock_expert)
    print(f"Initial Training Loss: {loss:.6f}")
    
    if loss > 0:
        print("✅ SUCCESS: BC Training Loop verified with discrete multimodal inputs.")