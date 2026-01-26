
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
touch docs/reports/evaluation_results.md

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
