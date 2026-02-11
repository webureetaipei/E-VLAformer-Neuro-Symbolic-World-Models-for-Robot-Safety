
# Environment Setup Guide

## 1. Prerequisite: NVIDIA Driver & WSL2 (Task 01)
Before starting, ensure your Windows host meets the following requirements:
- **OS:** Windows 11 with WSL2 enabled (Ubuntu 20.04/22.04).
- **GPU:** NVIDIA RTX GPU (Driver version **535.xx** or later).
- **Networking:** Ensure WSL2 allows localhost connections.
## Project Structure Initialization (Part of Task 01)
*Alignment: This step completes the "Project initialization" goal defined in Task 01.*

Run the following commands to create the standard Google-style research directory structure and initialize Python packages.

```bash
# 1. Create Core Source Directories (Modular Architecture)
# src/sim      -> Isaac Sim integration
# src/tiny_vla -> C++ Embedded Engine Port
# src/vla      -> Vision-Language-Action Models
# src/serving  -> gRPC Distributed Server
mkdir -p src/{serving,sim,tiny_vla,utils,vla}
mkdir -p configs tests docs/design docs/reports

# 2. Initialize Python Packages (Make directories importable)
touch src/__init__.py
touch src/serving/__init__.py
touch src/sim/__init__.py
touch src/tiny_vla/__init__.py
touch src/utils/__init__.py
touch src/vla/__init__.py

# 3. Create System Design Documentation (The "Google Design Docs")
touch docs/design/{system_design_overview.md,embedded_system_design.md,distributed_system_design.md,neuro_symbolic_multimodal_system_design.md}
touch docs/reports/evaluation_results.mdgit status

# 4. Create Git Keepfiles & Root Docs
touch configs/.gitkeep tests/.gitkeep
# Note: ROADMAP.md and setup_guide.md are typically at the project root
touch ROADMAP.md setup_guide.md README.md

echo "✅ Project structure created successfully."
```

## 2.Docker & Isaac Sim Installation (Task 02)
We utilize the official NVIDIA container for simulation.
```bash
# Install Docker & NVIDIA Toolkit
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-container-toolkit

# Pull Image
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
```
## 3. Python & CUDA Environment Setup (Task 03)
We use Miniconda to manage the deep learning environment.
```bash
# 4.1 Create Environment
# Create environment from file
conda env create -f environment.yml
conda activate evlaformer

# 4.2 Verification (Milestone Check)
python check_gpu.py
```
**Expected Output:** ✅ CUDA Available: True


## 4. Headless Simulation Configuration (Task 04)
Target: Run simulation on remote servers without GUI.
```bash
# To run Isaac Sim in headless mode (for training/data generation):
# Basic headless run command
docker run --name isaac-sim --entrypoint ./runheadless.native.sh --gpus all -e "ACCEPT_EULA=Y" --rm -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    nvcr.io/nvidia/isaac-sim:4.2.0
```
**Expected Output:** ✅ You should see a stream of "Omniverse Kit" startup logs indicating the simulation server is running.

## 5. gRPC Communication Verification (Task 05)
Target: Establish the bidirectional command-response loop between Client and Sim Server.

```bash
# 1. Start Isaac Sim (Windows Native or Docker Headless)
# 2. Run the gRPC verification script
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\generate_data.py
```

## 6. HDF5 Data Engine & Validation (Task 06)
Target: High-performance data logging and visual alignment check.
```bash
6.1 Install Dependencies
# Use the Isaac Sim python wrapper to install h5py
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat -m pip install h5py

6.2 Generate & Verify Dataset

# 1. Generate HDF5 dataset
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\generate_data.py

# 2. Run automated visual validation
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\check_hdf5.py
```
**Expected Output:**

✅ dataset_v1.hdf5 created in data/output/.

✅ A popup window showing RGB and Semantic Mask in pixel-perfect alignment.

✅ Standardized plot saved to docs/images/visual_validation.png.

## 7. Causal Event Labeling & Physics Validation (Task 07)
Target: Synchronize physics contact/proximity events with multimodal HDF5 streams.
```bash
7.1 Physics Simulation & Event Capture
This step simulates a multi-object interaction and automatically injects "Collision" flags into the metadata when a specific proximity threshold is met.
# Execute the collision simulation script
# This script initializes the World, spawns interacting objects, 
# and logs synchronized RGB + Physics Metadata.
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\generate_data.py

7.2 Automated Event Verification
Run the event-aware validation utility. This script scans the HDF5 file for the collision_event == True flag and extracts the exact "Impact Frame" for visual inspection.
# Run the causal event visualizer
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\check_hdf5.py
```
**Expected Output:**

✅ collision_data.hdf5 generated with synchronized collision_event boolean array.

✅ Terminal Log: 💥 Collision detected at Frame X! (verifying physics-to-metadata sync).

✅ Validation Plot: collision_validation.png saved to docs/images/, showing the visual impact frame with overlaid causal metadata.

## 8. Domain Randomization (DR) & Dataset Diversity (Task 08)
Target: Implement automated visual and physical variance to prevent model overfitting.
```bash
8.1 Randomized Data Generation
This step generates a dataset where object colors, lighting conditions, and physical properties (mass) vary across every simulation run.
# Run the randomization engine
# This script applies rep.randomizer.color and randomizes mass values
# while maintaining synchronized causal labeling.
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\generate_data.py

8.2 DR Pipeline Validation
Verify that the randomization logic is correctly reflected in the HDF5 metadata and that the visuals are rendering with the expected variety.
# Run the DR validation utility
# Scans for randomized_data.hdf5 and extracts a sample impact frame
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\check_hdf5.py
```
**Expected Output:**
✅ randomized_data.hdf5 created with unique physical_props (mass_a, mass_b) in the metadata stream.

✅ Validation Plot: randomization_validation.png saved to docs/images/.

✅ Visual Confirmation: Objects appear in randomized colors (not default grey) with varied lighting intensity.

## 9. Dataset Auditing & Unit Testing (Task 09)
Target: Automated verification of HDF5 data integrity and causal metadata accuracy.
```bash
9.1 Execute Data Audit
This script performs a "Stress Test" on the generated randomized_data.hdf5 to ensure no "dead frames" (black/white pixels) or corrupted metadata exist.
# Run the automated dataset auditor
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\utils\audit_dataset.py

9.2 Audit Checklist
The auditor validates the following critical research requirements:

- Structural Integrity: Verifies that rgb, collision_event, and metadata datasets are correctly indexed.

- Visual Entropy: Ensures RGB frames contain valid visual information (checks mean pixel density).

- Causal Consistency: Confirms that collision_event flags are synchronized with physical proximity.

- Metadata Parsability: Validates that JSON-encoded physical_props (mass, colors) are within the Domain Randomization (DR) bounds.
```
**Expected Output:**

🔍 Starting Dataset Audit...

  ✅ Dataset 'rgb' found.

  ✅ RGB Integrity: Mean pixel value 128.45

  ✅ Causal Check: Found X collision frames.

  ✅ Physics Check: Mass_A (X.XXkg) within DR bounds.
  
⭐ AUDIT COMPLETE: Dataset is certified for Phase 2 Training.

## 10. High-Entropy Batch Generation (Task 10)
Target: Scaling data production to create a diversified research dataset for GNN training.
```bash
10.1 Dataset Scaling Strategy
Due to Windows subprocess environment constraints, we utilize Manual Batch Scaling. This ensures the Isaac Sim environment is correctly initialized for every data sequence while allowing for high visual and physical entropy.

10.2 Generation Workflow

1.Open src/sim/generate_data.py.

2.Update the FILE_ID variable for the current run (e.g., "001", "002", etc.).

3.Execute the generator:
C:\Users\Michael\Desktop\isaac-sim-standalone-4.5.0-windows-x86_64\python.bat src\sim\generate_data.py

4.Repeat until the desired dataset size is reached in data/output/batch_v1/.

10.3 Verification of Scaling
The batch generation is successful when the following directory structure is populated:

- data/output/batch_v1/sim_data_batch_001.hdf5

- data/output/batch_v1/sim_data_batch_002.hdf5

- data/output/batch_v1/sim_data_batch_003.hdf5

Each file contains unique Domain Randomization samples (varying mass, colors, and light intensity) but shares a standardized HDF5 schema.
```

# Task 11: Graph Data Infrastructure Setup

This guide documents the environment configuration and verification for the Object-Centric Dataset Loader.

This guide documents the environment configuration and verification for the Object-Centric Dataset Loader.

## 1. Environment Requirements
The project utilizes **WSL2 (Ubuntu)** with a dedicated Conda environment.

### Core Dependencies
* **Python:** 3.10
* **PyTorch:** Installed via Pip (CPU version) to ensure compatibility with WSL2 symbols.
* **Torch Geometric:** For Graph Neural Network data structures.
* **h5py:** For memory-efficient streaming of large dataset chunks.

## 2. Installation Steps
```bash
conda activate evla
# Ensure PyTorch is compatible with WSL2
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
# Install Graph and Data libraries
pip install torch-geometric h5py requests
```


