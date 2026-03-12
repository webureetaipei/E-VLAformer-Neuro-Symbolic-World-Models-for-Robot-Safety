import matplotlib.pyplot as plt
import numpy as np

# 1. Your actual observed milestones (Epoch, Loss)
# I used the values from your screenshots:
milestones = [
    (1, 0.289045), 
    (5, 0.250827), 
    (15, 0.249885), 
    (25, 0.249754), 
    (45, 0.249592), 
    (80, 0.249514)
]

# 2. Extract X and Y
x_milestones, y_milestones = zip(*milestones)

# 3. Create 80 points using interpolation (Smoothing the curve)
epochs = np.arange(1, 81)
# Use 'pchip' or 'linear' interpolation to fill the gaps
losses = np.interp(epochs, x_milestones, y_milestones)

# 4. Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, color='#2ecc71', linewidth=2.5, label='Huber Loss (E-VLAformer)')

# Add vertical lines for the Learning Rate drops we saw in your logs
plt.axvline(x=19, color='red', linestyle='--', alpha=0.3, label='LR Drop (5e-05)')
plt.axvline(x=26, color='orange', linestyle='--', alpha=0.3, label='LR Drop (2.5e-05)')
plt.axvline(x=44, color='yellow', linestyle='--', alpha=0.3, label='LR Drop (3.13e-06)')

# Formatting for NeurIPS style
plt.title('Training Convergence: Behavioral Cloning Policy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Huber)', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend()

# 5. Save the Image
plt.savefig('training_convergence.png', dpi=300, bbox_inches='tight')
print("✅ Success! Your plot is saved as: training_convergence.png")