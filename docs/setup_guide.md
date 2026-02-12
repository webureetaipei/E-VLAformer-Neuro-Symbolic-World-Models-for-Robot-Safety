
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

## 11. Graph Data Infrastructure Setup (Task 11)

This guide documents the environment configuration and verification for the Object-Centric Dataset Loader.

This guide documents the environment configuration and verification for the Object-Centric Dataset Loader.

### 1. Environment Requirements
The project utilizes **WSL2 (Ubuntu)** with a dedicated Conda environment.

#### Core Dependencies
* **Python:** 3.10
* **PyTorch:** Installed via Pip (CPU version) to ensure compatibility with WSL2 symbols.
* **Torch Geometric:** For Graph Neural Network data structures.
* **h5py:** For memory-efficient streaming of large dataset chunks.

### 2. Installation Steps
```bash
conda activate evla
# Ensure PyTorch is compatible with WSL2
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
# Install Graph and Data libraries
pip install torch-geometric h5py requests
```
### 3. Data Architecture

The system transforms raw sensory tensors into a relational graph structure ($G = \{V, E\}$):

- Nodes (V): Feature vectors include [x, y, z, mass, type_id], allowing the model to distinguish between robot joints and environmental objects.

- Edges (E): Physical and spatial constraints stored in COO format (Coordinate format) for optimized GNN message passing.

- Metadata: * sim_path: Logical pointer to NVIDIA Isaac Sim prims for digital twin synchronization.

   - hw_id: Direct hardware mapping for Physical ESP32 (MG996R servos) during real-world execution.4. Verification

### 4. Verification
To ensure the environment and data pipeline are correctly configured, run the following smoke test to generate and load dummy graph data:
```bash
python src/models/graph_dataset.py
```

## 12. Relational Graph Construction (Task 12)
This guide documents the implementation and verification of the Graph Builder, which defines the structural and dynamic relationships between robot components and environment objects.

### 1. Functional Logic
The RelationalGraphBuilder (Task 12) implements two primary edge types to guide the model's causal reasoning:

- Kinematic Constraints: Establishes a permanent, bi-directional link between the Base, Joints, and Gripper for the 4-DOF DIY arm.

- Contact Detection: Dynamically instantiates edges when the Euclidean distance between the Gripper and a target object falls below a specified threshold (Default: $0.05m$).

### 2. Implementation File
The core logic is stored in src/models/graph_builder.py. This module acts as the intermediate layer between raw data loading and the Graph Neural Network (GNN).

### 3. Verification & Integration Test
Because this task involves complex spatial logic, an integration test is required to ensure that the "Nervous System" (Task 11) correctly passes data to the "Relational Logic" (Task 12).
```bash
To run the Task 12 verification:
conda activate evla
# Run the integration test from the project root
python -m src.utils.verify_task12
```
**Expected Output:**

--- Phase 2: Relational Graph Construction ---
Total Edges Found: 8

✅ Task 12 SUCCESS: Kinematic and Contact edges generated.

## 13. GNN Message Passing Infrastructure (Task 13)
This guide documents the implementation of the inductive reasoning processor.

### 1. Functional Logic
The `EVLAGNNProcessor` utilizes **GraphSAGE (SAGEConv)** layers to perform feature aggregation across the relational graph.
- **Architecture:** 3-layer GNN with 64 hidden channels and 32-dim output embeddings.

### 2. Verification
Run the forward pass test to verify neural data flow:
```bash
conda activate evla
export PYTHONPATH=$PYTHONPATH:.
python -m src.utils.verify_task13
```
## 14. Multimodal Fusion Infrastructure (Task 14)
This guide outlines the technical requirements and implementation steps for fusing the Vision-Language features with the Graph World Model (GWM) outputs.

### 1. Functional Objectives

Gemini said
Task 14 marks the beginning of the Multimodal Fusion Layer, the critical component that aligns and integrates diverse sensory streams—vision, language, and the GNN-based physical embeddings you just built. This layer serves as the "Global Workspace" where abstract instructions and raw physical data merge into a unified representation for action prediction.

Task 14: Multimodal Fusion Infrastructure
This guide outlines the technical requirements and implementation steps for fusing the Vision-Language features with the Graph World Model (GWM) outputs.

- Functional Objectives
The primary goal is to perform Intermediate Fusion to bridge the "Reality Gap" between visual pixels and symbolic physical constraints.
- Feature Alignment: Aligning high-frequency proprioception and graph embeddings with lower-frequency visual tokens.
- Semantic Mapping: Using a Q-Former style query mechanism to extract relevant physical features from the GNN based on the textual instruction (e.g., "pick up the red cube").
### 2. Technical Architecture
The MultimodalFusionLayer utilizes Cross-Attention to allow the visual-language backbone to "interrogate" the Graph World Model for physical consistency.
- Inputs:
    - Visual Tokens ($\mathbf{T}_{vis}$): From the ViT/CLIP encoder.
    - Instruction Tokens ($\mathbf{T}_{lang}$): From the language backbone.
    - GNN Embeddings ($\mathbf{T}_{graph}$): 32-dimensional physical latent vectors from Task 13.
- Mechanism:
    - Query ($Q$): Generated from language and vision tokens to focus on task-relevant objects.
    - Key ($K$) & Value ($V$): Derived from the GNN embeddings to provide physical context (mass, position, hierarchy).
### 3. Verification & Integration Test
A successful Task 14 requires verifying that the fused embedding contains information from all three sources.
```bash
#Run the verification script (to be created as src/utils/verify_task14.py):
conda activate evla
export PYTHONPATH=$PYTHONPATH:.
python -m src.utils.verify_task14
```
**Expected Output:**

--- Task 14: Multimodal Fusion Verification ---

Input Vision Tokens: torch.Size([1, 256, 512])

Input GNN Embeddings: torch.Size([4, 32])

Fused Multimodal Embedding: torch.Size([1, 256, 512])

✅ Task 14 SUCCESS: Multimodal alignment and cross-attention verified.

## 15. Latent Space Visualization Setup (Task 15)

### 1. Additional Dependencies
Task 15 requires scikit-learn for manifold learning and seaborn for high-fidelity scientific plotting.
```bash
# Install visualization stack
pip install scikit-learn seaborn matplotlib
```
### 2. Directory Preparation

The visualization script exports high-resolution PNGs to the reports directory. Ensure the path exists:
```bash
mkdir -p docs/reports
```
### 3. Execution Protocol

To generate the t-SNE manifold of the GNN latent space, run the utility from the project root. This command ensures the src package is correctly indexed in your PYTHONPATH.

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m src.utils.visualize_graph_latents
```

**Expected Output:**
- Console: You should see 🔄 Processing X samples... followed by 📉 Computing t-SNE....
- Artifact: A file named gnn_latent_clusters.png will be generated in docs/reports/.

## 16. Supervised Contrastive Training (Task 16)

This guide details the implementation of the **Supervised Contrastive Learning** pipeline. This process transforms the GNN from a random feature extractor into a physical world model capable of distinguishing between "Safe" and "Collision" states.

---

### 1. Environment & Dependencies
Ensure your "Graph AI" stack is installed in your WSL2/Conda environment:

```bash
# Core Machine Learning
pip install torch torchvision torchaudio
# Graph Neural Networks
pip install torch-geometric
# Data Handling
pip install h5py
```

### 2. Training Strategy

Unlike standard classification, we use InfoNCE (Information Noise Contrastive Estimation) to shape the latent manifold.

- Positive Pairs: Successive frames in a "Safe" trajectory should be pulled together.

- Negative Pairs: A "Safe" state and a "Collision" state must be pushed apart.

- Objective: Maximize the Manifold Expansion Ratio to create clear safety boundaries.

### 3. Execution Workflow
Once the dependencies are verified, follow these steps to ground your model:

- Initialize Weights: Ensure models/weights/ exists to store the .pth artifacts.

- Run Trainer: Execute python src/models/train_gnn_contrastive.py.

- Audit Topology: Run the t-SNE visualizer to verify the axis expansion from 150 to 400 units.