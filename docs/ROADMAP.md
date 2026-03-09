# E-VLAformer Project Roadmap: The 100 Tasks to NeurIPS

**Status:** Active
**Goal:** NeurIPS 2026
**Core Tech:** TinyEngine (C++), Graph World Model, Isaac Sim, Distributed gRPC

---

## Phase 1: Infrastructure & Environment (Tasks 01-10)
- [x] **Task 01:** Project initialization, Git setup, and Technical Spec v2.1.
- [x] **Task 02:** Deploy Isaac Sim (Docker/Local) and verify "Hello Robot".
- [x] **Task 03:** Setup AI Environment (PyTorch & CUDA) and prepare for C++.
- [x] **Task 04:** Configure Docker container for Headless Simulation.
- [x] **Task 05:** Verify gRPC communication loop (Python Client <-> Sim Server).
- [x] **Task 06:** Setup HDF5 data logger infrastructure.
- [x] **Task 07:** Physics Event: Multi-Object Collision & Causal Labeling.
- [x] **Task 08:** Domain Randomization: Automated Material & Visual Variance.
- [x] **Task 09:** Unit Testing: Dataset Integrity & Physics Consistency Checks.
- [x] **Task 10:** Phase 1 Review: Finalize Sim-to-Real Data Pipeline Documentation.
## Phase 2: Graph World Models (Tasks 11-20)
- [x] **Task 11:** HDF5-to-Graph Dataset Loader (Object-Centric Representation) 
      - *Outcome:* Verified HDF5-to-Graph pipeline with lazy-loading and metadata support.
- [x] **Task 12:** Construct the Relational Graph (Nodes: Prims, Edges: Physics Constraints).
      - *Outcome:* Automated generation of Kinematic and Dynamic Contact edges for the 4-DOF DIY arm.
- [x] **Task 13:** Implement Graph Convolutional Layers (PyTorch Geometric).
      - *Outcome:* Built 3-layer GraphSAGE processor for inductive physical reasoning.
- [x] **Task 14:** Multimodal Fusion Layer (Vision + Language + GNN).
      - *Outcome:* Verified Cross-Attention alignment between ViT tokens and GNN embeddings.

- [x] **Task 15:** Latent Space Topology & t-SNE Manifold Visualization. - Outcome: Successfully projected 32-dim physical embeddings into 2D space. Established baseline for topological consistency audit. - Artifact: docs/reports/gnn_latent_clusters.png

- [x] **Task 16:** Supervised Contrastive Grounding
   - Objective: Implement InfoNCE loss to maximize the mutual information between graph-encoded states and safety labels.
   - Outcome: Structured the latent manifold to distinguish between Safe Reach and Collision topologies, achieving a verified scale expansion from 150 to 400 units.
   - Artifact: gnn_contrastive_beta.pth.

- [x] **Task 17: Global State Persistence: Graph Memory Buffer.** 
    - **Logic:** Implemented a TTL-based (Time-To-Live) circular buffer for node persistence.
    - **Verification:** Successfully passed the "Lid Test" simulation; objects remain in the graph for **30+ frames** (extrapolatable to 1,000+ frames) during 100% occlusion.
    - **Metric:** Zero-drift latent retention confirmed via `verify_task17.py`.

- [x] **Task 18:** Edge Case Hardening: Handling Vanishing Nodes (Occlusion) 
    - *Outcome:* Integrated "Blink Logic" in Isaac Sim; verified 10% occlusion resilience in HDF5 stream.

- [x] **Task 19:** Silhouette Audit: Quantifying Latent Stability 
    - *Outcome:* Achieved 0.00 Silhouette Deviation (Identity Mapping) with 0.53 feature variance.
- [x] **Task 20:** Phase 2 Technical Review & GWM Performance Freeze 
    - *Decision:* Model is stable; proceeding to Action Policy.

    ## Phase 3: Multimodal VLA Policy & Action (Tasks 21-40)
*Status: 🚀 Initiating | Focus: Decision Making & Trajectory Generation*

### 🧠 Task 21-25: Policy Architecture
- [x] **Task 21:** Implement the **VLA Transformer Head**: Multi-layer MLP/Transformer for action prediction.
- [x] **Task 22:** **Joint Space Proprioception**: Integrate real-time encoder feedback ($\theta_{1-4}$) into the Policy input.
- [x] **Task 23:** **Language Grounding**: Implement a CLIP-based text encoder to process commands (e.g., *"Pick up the red cube"*).
- [x] **Task 24:** **Action Tokenization**: Convert raw joint deltas into discrete tokens for multi-modal alignment.
- [x] **Task 25:** **Safety Constraint Layer**: Integrate the CRM (Causal Reasoning Module) to mask "impossible" actions.

### 🦾 Task 26-30: Expert Demonstration & Behavioral Cloning
- [x] **Task 26: Kinesthetic Scripting (Verified)**: Develop Isaac Sim script to generate "Expert" trajectories (Pushing/Reach).
- [x] **Task 27: Demonstration Harvesting (Verified)**: Generate high-entropy episodes with Domain Randomization and Automated Movement Auditing.
- [x] **Task 28: Pick-and-Place State Machine (Verified)**: Upgraded the controller to a Multi-Phase Logic (Hover -> Dive -> Grasp -> Lift -> Place). Integrated the **"Iron Grip" physics protocol** and certified 5-scenario H5 data coverage.
- [x] **Task 29: Robustness Production (Verified)**: Scaled the pipeline to generate the **"Robustness Trinity"** dataset (50 episodes). Verified 100% integrity across Normal, Occlusion (Object Permanence), and Perturbation (Reactive) scenarios.

| Normal Baseline | Visual Occlusion | Dynamic Perturbation |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/51a7333a-c138-4abe-b1b0-2d84bc367c40" width="100%"></video> | <video src="https://github.com/user-attachments/assets/0be5fa49-3628-478a-9c24-0fe5bb534e3d" width="100%"></video> | <video src="https://github.com/user-attachments/assets/b4f42cc8-ef76-4212-956c-a51b0f0ff03e" width="100%"></video> |

- [ ] **Task 30: VLA Dataset & Training Pipeline**: Build the PyTorch `DataLoader` to aggregate H5 trajectories. Implement joint normalization and initiate the first Behavioral Cloning training loop.

### 🧪 Task 31-35: Long-Horizon Reasoning
- [ ] **Task 31: Multi-Stage Sequences**: Implement "Unstack → Move → Restack" causal workflows using the Persistent GWM.
- [ ] **Task 32: Graph-Guided Search**: Use the GNN to predict intermediate sub-goals for complex, multi-object tasks.
- [ ] **Task 33: Recovery Policies**: Train the model to "re-try" or "re-acquire" if the cube is dropped or grippers slip.
- [ ] **Task 34: Hallucination Audit**: Verify the Policy doesn't "reach for shadows" during occlusion (Task 19 Identity Validation).
- [ ] **Task 35: Hierarchical VLA**: Formally separate High-level "Intent" from Low-level "Joint Control" streams.

### 🔬 Task 36-40: Optimization & Technical Freeze
- [ ] **Task 36: Inference Latency Profile**: Measure End-to-End time from Pixel input to PWM output in Python/WSL2.
- [ ] **Task 37: Policy Quantization (PTQ)**: Prepare weights for Int8 conversion and TinyEngine (C++) compatibility.
- [ ] **Task 38: Dataset Scaling**: Expand to 10,000+ synthetic episodes with extreme Domain Randomization (DR).
- [ ] **Task 39: Phase 3 Documentation**: Draft the "Action Policy Methodology" section for the NeurIPS paper.
- [ ] **Task 40: VLA Performance Freeze**: Lock the Action-Head architecture and weights before Phase 4 (Embedded Porting).
---
*Note: This roadmap is a living document. We will check off tasks as we complete them.*