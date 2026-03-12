
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

This guide documents the environment configuration and verification for the Object-Centric Dataset Loader and the raw-to-graph preprocessing pipeline.

### 1. Environment Requirements
The project utilizes **WSL2 (Ubuntu)** with a dedicated Conda environment.

#### Core Dependencies
* **Python:** 3.10
* **PyTorch:** Installed via Pip (CPU version) to ensure compatibility with WSL2 symbols.
* **Torch Geometric:** For Graph Neural Network data structures and message passing.
* **h5py:** For memory-efficient streaming of large dataset chunks from the Master HDF5.

### 2. Installation Steps
```bash
conda activate evla
# Ensure PyTorch is compatible with WSL2
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
# Install Graph and Data libraries
pip install torch-geometric h5py requests
```
### 3. Data Architecture & Preprocessing
The infrastructure utilizes a two-stage pipeline to transform raw sensory tensors into a relational graph structure ($G = \{V, E\}$):
#### A. Preprocessing (src/data/preprocess_raw_to_graph.py)
This utility serves as the bridge between the HDF5 Data Engine and the GNN. It extracts synchronized state vectors and formats them for the Graph Builder.
- Input: Raw HDF5 Trajectories (RGB, Joint States, Object Poses).
- Operation: Flattens multimodal streams into object-centric feature tensors.
- Output: Intermediate Graph Tensors ready for edge instantiation.
#### B. Graph Structure
- Nodes (V): Feature vectors include $[x, y, z, m, \mathrm{type\_id}]$, allowing the model to distinguish between robot joints and environmental objects.
- Edges (E): Physical and spatial constraints stored in COO format (Coordinate format) for optimized GNN message passing.
- Metadata Alignment: - sim_path: Logical pointer to NVIDIA Isaac Sim prims for digital twin synchronization.
  - hw_id: Direct hardware mapping for Physical ESP32 (MG996R servos) during real-world execution.
### 4. Verification
To ensure the environment and the preprocessing pipeline are correctly configured, run the following smoke tests:
Step 1: Verify Preprocessing Logic
```bash
# Test the raw data to graph tensor conversion
python src/data/task11_preprocess_raw_to_graph.py --test_mode
```
Step 2: Verify Graph Loading
```bash
# Test the Geometric Dataset Loader and COO format integrity
python src/models/graph_dataset.py
```
✅ Task 11 Certified: Environment is stable, and the preprocess_raw_to_graph.py utility is confirmed to handle 548-dim multimodal synchronization.
## 12. Relational Graph Construction (Task 12)
This guide documents the implementation and verification of the **Graph Builder**, which defines the structural and dynamic relationships between robot components and environment objects.

### 1. Functional Logic (`src/models/task12_graph_builder.py`)
The `RelationalGraphBuilder` implements the symbolic logic required to transform spatial coordinates into a relational topology. It defines two primary edge categories:

#### A. Kinematic Constraints ($E_{kin}$)
Establishes permanent, bi-directional links based on the physical assembly of the **Franka Emika Panda**.
* **Topology:** `Base -> Link1 -> Link2 -> ... -> Hand -> Gripper`.
* **Purpose:** Ensures the GNN understands that the movement of the base link propagates through the entire kinematic chain.

#### B. Dynamic Contact Detection ($E_{con}$)
Instantiates edges in real-time based on proximity sensors or Euclidean distance.
* **Threshold:** Default is set to **$0.05m$**.
* **Trigger:** If $dist(\text{Gripper}, \text{Object}) < 0.05$, a "Contact Edge" is created, allowing the VLA model to reason about grasping and object manipulation.

### 2. Implementation Overview
The core logic resides in `src/models/task12_graph_builder.py`. This module acts as the intermediate reasoning layer between the `preprocess_raw_to_graph.py` (Task 11) and the GNN Processor (Task 13).



### 3. Verification & Integration Test
This test ensures that the "Nervous System" (Task 11) correctly feeds the "Relational Logic" (Task 12) without spatial data corruption.

```bash
# Activate environment
conda activate evla

# Run the Task 12 verification (Checks for adjacency matrix integrity)
python -m src.utils.verify_task12
```
**Expected Output:**

--- Phase 2: Relational Graph Construction ---
Total Edges Found: 8

✅ Task 12 SUCCESS: Kinematic and Contact edges generated.

## 13. GNN Message Passing Infrastructure (Task 13)
This guide documents the implementation of the inductive reasoning processor, which serves as the "Spatial Brain" of the E-VLAformer.

### 1. Functional Logic (`src/models/task13_gnn_processor.py`)
The `EVLAGNNProcessor` is responsible for performing inductive reasoning over the relational graph built in Task 12. It utilizes **GraphSAGE (SAGEConv)** layers to aggregate features from neighboring nodes (e.g., propagating joint torque information to the gripper node).

#### Architecture Specifications:
- **Backbone:** 3-layer GraphSAGE Architecture.
- **Hidden Dimensions:** 64 channels per layer with **GELU** activation.
- **Output Latent:** 32-dimensional embedding representing the unified physical world-state.
- **Inductive Bias:** The processor is designed to handle variable graph sizes, ensuring compatibility if additional objects are added to the simulation scene.



### 2. Implementation Details
The core model definition is located in `src/models/task13_gnn_processor.py`. This script defines the `Forward` pass logic, including:
1. **Neighborhood Aggregation:** Gathering $[x, y, z, m]$ features from connected links.
2. **Feature Fusion:** Updating node hidden states based on kinematic constraints.
3. **Global Pooling:** (Optional) Generating a graph-level summary for the VLA Policy Head.

### 3. Neural Data Flow Verification
Run the forward pass test to verify that the GNN can ingest the COO-format edges from Task 12 and produce a non-zero 32-dim latent:

```bash
# Activate environment and set path
conda activate evla
export PYTHONPATH=$PYTHONPATH:.

# Execute the GNN forward-pass smoke test
python -m src.utils.verify_task13
```
**Expected Output:**

--- 🧠 Phase 2: GNN MESSAGE PASSING AUDIT ---

[GNN] Ingesting Graph (Nodes: 8, Edges: 14)

[GNN] Forward Pass: 3-Layer SAGEConv...

✅ Tensor Shape: [1, 32] (Physical Latent Verified)

✅ Gradient Path: Functional

✅ Latent Variance: 0.5309 (Rich Feature Extraction)

🚀 Task 13 SUCCESS: The GNN Processor is certified for Neuro-Symbolic Reasoning.
## 14. Multimodal Fusion Infrastructure (Task 14)
This guide outlines the technical requirements and implementation steps for fusing Vision-Language features with the Graph World Model (GWM) outputs.

### 1. Functional Objectives (`src/models/fusion_layer.py`)
The primary goal is to perform **Intermediate Fusion** to bridge the "Reality Gap" between visual pixels and symbolic physical constraints. The `MultimodalFusionLayer` serves as the global workspace where abstract instructions and raw physical data merge.

* **Feature Alignment:** Aligning high-frequency proprioception (100Hz) and graph embeddings with lower-frequency visual tokens (30Hz).
* **Semantic Mapping:** Utilizing a Cross-Attention mechanism to allow the VLA backbone to "interrogate" the GNN for physical consistency based on text instructions (e.g., *"Pick up the red cube"*).
* **Dimensionality Synchronization:** Compiling disparate streams into the certified **548-dim Fusion Vector** (32 GNN + 4 Joint + 512 Lang).

### 2. Technical Architecture
The `fusion_layer.py` implements a synchronized bottleneck where vision and language tokens are grounded by physical graph latents.

* **Inputs:**
    * **Visual Tokens ($\mathbf{T}_{vis}$):** Extracted from the ViT/CLIP encoder.
    * **Instruction Tokens ($\mathbf{T}_{lang}$):** 512-dim embeddings from the Language Handler (Task 23).
    * **GNN Embeddings ($\mathbf{T}_{graph}$):** 32-dim physical latent vectors from Task 13.
    * **Proprioception:** 4-dim normalized joint states from Task 22.
* **Mechanism:**
    * **Query ($Q$):** Generated from language/vision to focus on task-relevant nodes.
    * **Key ($K$) & Value ($V$):** Derived from the GNN embeddings to provide physical context (mass, position, hierarchy).



### 3. Verification & Integration Test
This test verifies that the `fusion_layer.py` correctly concatenates and aligns all three sources without data loss or shape mismatch.

```bash
# Activate environment and set path
conda activate evla
export PYTHONPATH=$PYTHONPATH:.

# Run the Task 14 verification
python -m src.utils.verify_task14
```
**Expected Output:**
--- 🧠 Task 14: MULTIMODAL FUSION VERIFICATION ---

[Fusion] Ingesting Language Latents: torch.Size([1, 512])

[Fusion] Ingesting GNN Physical Latents: torch.Size([1, 32])

[Fusion] Ingesting Joint Proprioception: torch.Size([1, 4])

✅ Total Fusion Vector Dim: 548 

✅ Attention Map Sparsity: Verified

✅ Modality Alignment: SUCCESS

🚀 Task 14 SUCCESS: Multimodal alignment and cross-attention logic certified.

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
python -m src.utils.task15_visualize_graph_latents
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

- Run Trainer: Execute python src/models/task16_train_gnn_contrastive.py.

- Audit Topology: Run the t-SNE visualizer to verify the axis expansion from 150 to 400 units.

## 17. Global State Persistence (Graph Memory Buffer) (Task 17)

This guide details the implementation of the **Object Permanence** layer. This module allows the E-VLAformer to maintain a stable world state even when physical objects are occluded by containers (e.g., the "Lid Test") or move out of the camera's field of view.

---

### 1. Environment & Logic Requirements
Task 17 utilizes the existing "Graph AI" stack but introduces a TTL-based (Time-To-Live) caching mechanism.

- **Dependency:** `src/models/graph_memory.py`
- **Core Logic:** Circular persistence buffer with confidence-weighted decay.



---

### 2. Persistence Strategy
Unlike standard VLA models that reset the world state every frame, the Graph Memory Buffer implements **Cognitive Anchoring**:

- **Sight Sighting:** When an object is seen, its `TTL` (Time-To-Live) is reset to maximum (e.g., 30 frames).
- **Occlusion State:** If the object is not detected in the current frame, the system retrieves the last known position and features from the buffer.
- **Node Decay:** Every frame an object remains hidden, its `TTL` decreases. Once `TTL=0`, the node is removed from the "Mental Map."
- **Geometric Stability:** Maintained through zero-drift latent caching during the "Hidden" period.

---

###  3. Execution & Verification Workflow
Follow these steps to verify that your robot has achieved Object Permanence.

#### Step 3.1: Initialize the Memory Class
Ensure your `src/models/graph_memory.py` is correctly defined with the circular update logic.

#### Step 3.2: Run the Persistence Stress Test
Execute the automated verification utility. This simulates a "Blink" event where a detected cube disappears for 4 frames and checks if the GNN maintains the node.

```bash
# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Run the Task 17 Verification Script
python -m src.utils.verify_task17
```

### 18. Hardened Dataset Generation (Occlusion Resilience) (Task 18)
To train the World Model for object permanence, we use "Blink Logic" to simulate sensor dropouts or physical occlusions.

**1. Verify Blink Logic (Unit Test)**
Run the simulator-free logic test to ensure the stochastic occlusion math is working:
```bash
python src/utils/task18_test_blink_generator.py
```
## 🔬 Model Validation & Certification (Phase 2 Freeze)

## 19. Silhouette Latent Audit (Identity Stability) (Task 19)
This audit calculates the mathematical overlap between "Visible" and "Occluded" states to ensure the memory buffer is stable.

### 1. Run the Real-Weight Audit**
Ensure you have trained the GNN with Identity Mapping (Task 16.1) before running this. This script loads `gnn_contrastive_beta.pth` and interrogates the latent manifold.
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/utils/task19_silhouette_audit.py
```
### 2. Interpret the Result
- Score ~0.00: ✅ Success (Identity Preservation). The model perceives the object identically regardless of visibility.

- Score > 0.50: 🟡 Marginal (Clustered). The model distinguishes sight from memory (possible drift).

- Score < 0.00: ❌ Fail. The manifold is collapsing or corrupted.

## 20. Phase 2 Technical Review & Performance Freeze (Task 20)
Before moving to the Action Policy (Phase 3), we must certify the Graph World Model.

### 1. Verify Weight Integrity
Check the variance of the trained embeddings to ensure the "Brain" is producing rich features:
```bash
python -c "import torch; from src.models.gnn_processor import EVLAGNNProcessor; model = EVLAGNNProcessor(5, 64, 32); model.load_state_dict(torch.load('models/weights/gnn_contrastive_beta.pth')); x = torch.randn(1, 5); out = model(x, torch.tensor([[0],[0]])); print('Variance:', torch.var(out).item())"
```
Requirement: Variance > 0.1.
###  2. Lock Model Weights
Archive the certified weights to prevent accidental overwriting during Phase 3:
```bash
mkdir -p models/archive/phase2_freeze
cp models/weights/gnn_contrastive_beta.pth models/archive/phase2_freeze/certified_gwm_v1.pth
```
### 3. Dataset Certification
Verify that the Hardened Dataset (task18_occlusion_test_001.h5) is indexed and ready for VLA Training:
```bash
python src/utils/audit_dataset.py --file data/raw/task18_occlusion_test_001.h5
```
Note: Once archived, certified_gwm_v1.pth becomes the frozen backbone for all Phase 3 Action Policy tasks.

---


## 21. VLA Policy Head Architecture (Task 21)
This task initializes the neural bridge between the stable World Model (Phase 2) and the robotic actuators.

### 1. Implementation: Policy Head**
Create or verify the core architecture in `src/models/task21_vla_policy_head.py`. This Residual MLP is designed to fuse multimodal latents into normalized joint deltas.

### 2. Verify Architecture & Forward Pass**
Run the integration smoke test to ensure the dimensions for GNN (32), Proprioception (4), and Language (512) are correctly aligned.

```bash
# Activate Phase 3 Environment
conda activate evla

# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Execute Smoke Test
python src/models/task21_vla_policy_head.py
```


## 22. Joint Space Proprioception (Task 22)
This task establishes the mathematical bridge between raw sensor data (Degrees/Radians) and the normalized latent space required for Transformer-based policy inference.

### 1. Implementation: Proprioception Handler**
Create or verify the handler in `src/vla/task22_proprioception_handler.py`. This module implements a low-pass alpha filter to prevent simulation jitter from destabilizing the Policy Head.

### 2. Verify Normalization & Safety Clamping**
Run the verification script to ensure raw joint angles are mapped correctly to the $[-1, 1]$ range and that the smoothing logic is active.

```bash
# Activate Phase 3 Environment
conda activate evla

# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Execute Proprioception Verification
python src/vla/task22_proprioception_handler.py
```

## 23. Language Grounding (Task 23)
This task integrates the high-level command layer, converting natural language instructions into the 512-dimensional latent space required for the VLA Policy Head.

### 1. Install NLP Dependencies
Ensure the `sentence-transformers` library is installed within the `evla` environment to support the DistilRoBERTa backbone.

```bash
conda activate evla
pip install sentence-transformers
```

### 2. Implementation: Language Handler
Create or verify the handler in src/vla/language_handler.py. This module includes the 768 → 512 Projection Layer to ensure dimensionality alignment with the Task 21 Policy Head.

### 3. Verify Semantic Alignment
Run the verification script to certify that text strings are successfully mapped to 512-dimensional unit tensors.
```bash
# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Execute Language Verification
python src/vla/task23_language_handler.py
```
### 4. Expected Verification Output
```bash
🧠 Loading Language Encoder: all-distilroberta-v1...
📏 Projection Layer Initialized: 768 -> 512

--- Task 23: Aligned Language Verification ---
Command: 'Pick up the red cube'
Final Embedding Shape: [1, 512]
✅ DIMENSION ALIGNMENT SUCCESS: True
```

### 5. Multimodal Integration Audit
- Model Backbone: all-distilroberta-v1 (Lightweight/Fast).

- Output Dimension: 512-dim (Projected).

- Fusion Check: Aligned for 548-dim concatenation (32 GNN + 4 Joint + 512 Lang).

Note: The projection layer weights are currently initialized via nn.Linear. These will be fine-tuned during Phase 3 Behavioral Cloning to align semantic meaning with physical trajectories.

## 24. Multimodal Inference Engine (Task 24)
This task establishes the **"Central Nervous System"** of the E-VLAformer by synchronizing the frozen World Model, the physical sensor handlers, and the semantic instruction encoders into a single deterministic execution loop.

### 1. Implementation: The Orchestration Logic
The system utilizes two primary scripts to manage the "pixels-to-actions" pipeline:

* **The Orchestrator (`src/vla/task24_inference_engine.py`):** Handles asynchronous stream management (Vision @ 30Hz, Proprioception @ 100Hz) and ensures timing-consistent delivery to the policy head.
* **The Fusion Core (`src/models/fusion_layer.py`):** Acts as the mathematical bottleneck. It ingests the disparate streams and performs the final **548-dimensional fusion**, mapping the 32-dim GNN latent, 4-dim Joint vector, and 512-dim Semantic embedding into a unified tensor.



### 2. The 548-Dim Fusion Protocol
The `fusion_layer.py` ensures that the Policy Head receives a grounded representation of the world.
- **Symbolic Grounding:** The 32-dim GNN latent enforces physical consistency (Object Permanence).
- **Physical Grounding:** The 4-dim Proprioception vector provides real-time "self-awareness" of the arm's configuration.
- **Instructional Grounding:** The 512-dim Language vector directs the attention mechanism toward the task goal.

### 3. Execute Multimodal Synchronization Test
Run the engine in "Smoke Test" mode to certify that all handlers are communicating correctly without dimensionality mismatches or latency spikes.

```bash
# Activate Phase 3 Environment
conda activate evla

# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Execute Inference Engine Verification
python src/vla/task24_inference_engine.py
```
### 3. Expected Verification Output
--- 🚀 Task 24: INFERENCE ENGINE LIVE SYNC ---

[Engine] GNN Stream: ACTIVE (Latent: 32-dim)

[Engine] Joint Stream: ACTIVE (Proprio: 4-dim)

[Engine] Lang Stream: ACTIVE (Embed: 512-dim)

🔗 Calling FusionLayer... 

✅ Result: 548-dim Fusion Vector [1, 548]

✅ Latency: 14.2ms (E2E)

🚀 Task 24 SUCCESS: Central Nervous System is synchronized. Ready for BC Training.

## 25. Behavioral Cloning Pipeline (Task 25)
This task implements the supervised learning framework (BC) required to map the 548-dimensional multimodal input space to expert-level motor trajectories.

### 1. Implementation: BC Trainer
Verify or create the training orchestrator in `src/vla/task25_bc_trainer.py`. This module handles the backpropagation logic and ensures that the **GNN**, **Proprioception**, and **Language** streams are correctly weighted during optimization.

### 2. Execute Gradient Path Verification
Run the trainer in "Smoke Test" mode to certify that the `MSELoss` is calculating correctly and that the model weights are capable of updating across the multimodal fusion layers.

```bash
# Activate Phase 3 Environment
conda activate evla

# Set PYTHONPATH to project root
export PYTHONPATH=$PYTHONPATH:.

# Execute BC Training Verification
python src/vla/task25_bc_trainer.py
```

### 3. Expected Verification Output

🛠️ Testing Task 25: Behavioral Cloning Pipeline...
✅ Policy Head initialized with keyword arguments.
🏋️ BC Trainer Initialized on cuda (or cpu)

--- Task 25: Behavioral Cloning Verification ---
Initial Training Loss: 0.842109 (values will vary)
✅ SUCCESS: BC Training Loop verified with discrete multimodal inputs.

## 26. Synthetic Data Harvesting Engine (Task 26)
This task establishes the high-speed data generation pipeline in Isaac Sim, ensuring that visual renders are perfectly synchronized with robot joint states in the HDF5 buffer.

### 1. Implementation: Data Harvester
Verify the harvester logic in `src/data/task26_expert_harvester.py`. This module manages the `SimulationApp` lifecycle and coordinates the Replicator API to capture non-frozen RGB frames.

### 2. Execute Harvesting Verification
Run the harvester to capture a sample episode. This test confirms that the renderer is correctly drawing the robot's motion and saving it to the expert buffer.

```bash
# Execute Expert Data Harvesting
python src/data/task26_expert_harvester.py
```
### 3. Expected Verification Output

🛠️ Testing Task 26: Data Harvesting Engine...
--- 
🎬 Starting Episode 0 ---

📊 DATA AUDIT: Max Pixel = 255

💾 Successfully written episode_0 to Disk.

✅ SUCCESS: Data Engine verified. 150 frames captured and synchronized.

## 27. Domain Randomization & Quality Audit (Task 27)
This task certifies the dataset for AI training by enforcing environmental entropy and mathematically auditing the "Action Quality" of the captured sequences.

### 1. Implementation: Randomization & Audit
Ensure the randomize_scene and pixel_audit functions are active. These prevent overfitting by varying colors/positions and prevent "Empty Data" by checking for pixel movement.
### 2. Execute Entropy & Audit Verification
Run the randomized harvesting script. The system will automatically perform a Mean Absolute Difference (MAD) test to confirm the robot actually moved.
```bash
# Execute Randomized Harvesting with Auto-Audit
python src/data/expert_harvester_randomized.py
```
### 3. Expected Verification Output
--- 🎬 Starting Episode 0 | Target: [0.45, -0.12, 0.05] ---

📊 DATA AUDIT: Max Pixel = 255

✅ MOVEMENT VERIFIED: Pixel change score = 11.25

💾 Successfully written episode_0 to Disk.

✅ SUCCESS: Task 27 Certified. Dataset audited for high-entropy movement.

## 28. Physics Engineering & Scenario Coverage (Task 28)
This task certifies the physical reliability of the expert demonstrations and expands the dataset to include complex recovery scenarios (Obstacles and Collisions).

### 1. Implementation: "Iron Grip" & Scenario Logic
Ensure the `IronGrip` protocol is active in the controller. This forces negative joint commanding to prevent object slippage during high-acceleration maneuvers. Verify the `ScenarioManager` is configured to toggle between Normal, Obstacle, and Collision modes.

### 2. Execute Multi-Scenario Harvesting
Run the advanced state-machine script. The system will cycle through the 5 defined scenarios to ensure the HDF5 buffer contains "Out-of-Distribution" (OOD) recovery data.
```bash
# Execute Multi-Scenario Harvesting with Iron Grip
python src/data/task28_expert_grabber_scenarios.py
```
### 3. Expected Verification Output

--- 🚀 Starting Task 28: HEAVY GRIP MODE ---
[Lula] Joint mimic attributes ignored.

📊 SCENARIO: Obstacle (阻擋) | Object: [0.40, 0.10, 0.035]

🎯 ALIGNED. Dropping to grab...

✊ GRABBING (IRON GRIP): Commanding [-0.01, -0.01]

🚚 MOVING TO DROP ZONE (STABLE) | Vision: UNFROZEN (READY)

✅ SUCCESS: Task 28 Certified. Dataset verified for Iron-Grip stability and 5-Scenario coverage.

## 🌪️ Robustness Production & Stochastic Data Scaling (Task 29)
This task focuses on mass-producing high-entropy trajectories and verifying the **"Robustness Trinity"** (Normal, Occlusion, Perturbation) to ensure the E-VLAformer handles real-world failures and dynamic environments.

### 1. Implementation: The Robustness Pipeline
The data engine utilizes a suite of specialized scripts to ensure data diversity and integrity:
* `task29_collect_data_manager.py`: Orchestrates parallelized Isaac Sim sessions and robustness triggers.
* `task29_expert_harvester_randomized.py`: Executes the stochastic expert policy with Domain Randomization.
* `task29_check_h5_data.py`: Performs frame-by-frame auditing of visual and proprioceptive alignment.
* `task29_combined_h5.py`: Serializes individual episodes into the unified Master HDF5 structure.
* `task29_count_data.py`: Provides real-time distribution analytics across the Trinity categories.

### 2. Execute Expert Harvesting & Scaling
Run the randomized expert harvester to generate the 100-episode master batch. This applies Domain Randomization (DR) and specific robustness stressors.
```bash
# Execute Randomized Expert Harvesting for Task 29
python src/data/expert_harvester_randomized.py --episodes 100
```
### 3. Master Dataset Aggregation & Serialization
Once harvested, trajectories are audited and merged into the final training file for Phase 3.
```bash
# Verify data integrity and combine into master file
python src/data/task29_check_h5_data.py
python src/data/task29_combined_h5.py
python src/data/task29_count_data.py
```
### 4. Expected Verification Output

--- 🌪️ Starting Task 29: ROBUSTNESS PRODUCTION ---
[Isaac] Parallelizing environment sessions...

✅ Audit Complete: 100/100 files passed basic integrity.

📈 DISTRIBUTION: {'NORMAL': 33, 'OCCLUSION': 40, 'PERTURBATION': 27}

📦 DATASET: task30_training_master.h5 (Size: 1.20 GB)

👁️ OCCLUSION CHECK: "Blind Grasp" detected in Ep 42 | GWM Persistence: ACTIVE
🎯 PERTURBATION CHECK: Target Jump in Ep 88 | Reactive Path Correction: VERIFIED

✅ SUCCESS: Task 29 Certified. 100/100 episodes passed integrity audit. 
🚀 Master Dataset hosted: [https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset)

## 🧠 Unified Multimodal Training & Policy Convergence (Task 30)
This task focuses on the "Unmasked" optimization of the E-VLAformer brain, transitioning from static data harvesting to active policy learning across the multimodal manifold.

### 1. Implementation: The Training Pipeline
The training infrastructure leverages specialized modules to ensure gradient flow across the GWM and Vision streams:
* `task30_evla_dataset.py`: The "Unmasked" loader that pulls real-time GWM Graph Nodes and CLIP Language embeddings.
* `task30_train_vla.py`: The main optimization engine utilizing Huber Loss and `ReduceLROnPlateau` for precision motor control.
* `task30_plot_results.py`: A visualization utility that generates high-fidelity convergence plots for research audit.

### 2. Execute Policy Optimization
Run the unified training script to begin the 80-epoch optimization regime. This will dynamically adjust the Learning Rate as the model approaches the global minimum.
```bash
# Execute Elite Policy Training for Task 30
python src/vla/task30_train_vla.py
```

### 3. Convergence Visualization & Audit
Once training is complete, generate the convergence plot to verify the stability of the policy and the timing of the Learning Rate decays.
```bash
# Generate NeurIPS-standard training plot
python src/utils/task30_plot_results.py
```
### 4. Expected Verification Output
--- 🧠 Starting Task 30: UNIFIED POLICY CONVERGENCE ---

⚙️ Initializing SMART Training on: cuda

🚀 Commencing Advanced BC Optimization...
Epoch [01/80] | Loss: 0.289045 | LR: 1.00e-04
...
Epoch [19/80] | Loss: 0.249963 | LR: 5.00e-05 (Scheduler: Decay Triggered)
...
Epoch [80/80] | Loss: 0.249514 | LR: 1.22e-08

💾 Checkpoint Saved: models/weights/evla_advanced_epoch80.pth

📈 CONVERGENCE: Huber Loss stable at 0.2495

🧠 GWM INTEGRATION: Gradients active | Node-to-Action mapping: VERIFIED

✅ SUCCESS: Task 30 Certified. Elite Brain fully converged.
🚀 Weights Hosted: https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset

