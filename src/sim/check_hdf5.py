import h5py
import matplotlib.pyplot as plt
import os

def verify_and_save_proof():
    input_path = 'data/output/randomized_data.hdf5'
    output_image = 'docs/images/domain_randomization_proof.png'
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found!")
        return

    with h5py.File(input_path, 'r') as f:
        # Check frame 5 (usually stable after warm-up)
        img = f['rgb'][5]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title("Task 08: Domain Randomization Verification")
        plt.text(10, 30, f"Mean Pixel Val: {img.mean():.2f}", color='white', weight='bold')
        plt.axis('off')
        
        # Ensure target directory exists
        os.makedirs('docs/images', exist_ok=True)
        
        plt.savefig(output_image)
        print(f"✅ Success! Proof saved to: {output_image}")
        
        # Show plot if not in headless environment
        plt.show()

if __name__ == "__main__":
    verify_and_save_proof()