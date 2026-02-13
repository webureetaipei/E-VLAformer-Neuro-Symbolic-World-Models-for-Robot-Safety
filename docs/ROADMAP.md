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
- [ ] **Task 23:** **Language Grounding**: Implement a CLIP-based text encoder to process commands (e.g., *"Pick up the red cube"*).
- [ ] **Task 24:** **Action Tokenization**: Convert raw joint deltas into discrete tokens for multi-modal alignment.
- [ ] **Task 25:** **Safety Constraint Layer**: Integrate the CRM (Causal Reasoning Module) to mask "impossible" actions.

### 🦾 Task 26-30: Expert Demonstration & Behavioral Cloning
- [ ] **Task 26:** **Kinesthetic Scripting**: Develop Isaac Sim script to generate "Expert" trajectories (Reach & Grasp).
- [ ] **Task 27:** **Demonstration Harvesting**: Generate 500+ episodes of successful tasks with the Hardened GNN.
- [ ] **Task 28:** **Behavioral Cloning (BC) Training**: Train the Policy Head to mimic expert joint trajectories using MSE/Cross-Entropy.
- [ ] **Task 29:** **Rollout Testing**: Execute first autonomous simulation loop without hardcoded paths.
- [ ] **Task 30:** **Success Rate Benchmarking**: Record autonomous KPI (Success Rate vs. Path Smoothness).

### 🧪 Task 31-35: Long-Horizon Reasoning
- [ ] **Task 31:** **Multi-Stage Sequences**: Implement "Unstack → Move → Restack" causal workflows.
- [ ] **Task 32:** **Graph-Guided Search**: Use the GNN to predict intermediate sub-goals for complex tasks.
- [ ] **Task 33:** **Recovery Policies**: Train the model to "re-try" if the cube is dropped or grippers slip.
- [ ] **Task 34:** **Hallucination Audit**: Verify the Policy doesn't "reach for shadows" during occlusion (Task 19 validation).
- [ ] **Task 35:** **Hierarchical VLA**: Separate High-level "Intent" from Low-level "Joint Control."

### 🔬 Task 36-40: Optimization & Technical Freeze
- [ ] **Task 36:** **Inference Latency Profile**: Measure End-to-End time from Pixel to PWM in Python.
- [ ] **Task 37:** **Policy Quantization (PTQ)**: Prepare weights for Int8 conversion in TinyEngine.
- [ ] **Task 38:** **Dataset Scaling**: Expand to 10,000+ synthetic episodes with diverse Domain Randomization.
- [ ] **Task 39:** **Phase 3 Documentation**: Draft the "Methodology" section for the NeurIPS paper.
- [ ] **Task 40:** **VLA Performance Freeze**: Lock the Action-Head architecture before Phase 4 (C++ Porting).
---
*Note: This roadmap is a living document. We will check off tasks as we complete them.*