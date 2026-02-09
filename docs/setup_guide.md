
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


