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
*Status: 🟢 Core Architecture Verified | 🟡 Refining Logic*
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

- [x] **Task 17: Global State Persistence: Graph Memory Buffer.** ✅
    - **Logic:** Implemented a TTL-based (Time-To-Live) circular buffer for node persistence.
    - **Verification:** Successfully passed the "Lid Test" simulation; objects remain in the graph for **30+ frames** (extrapolatable to 1,000+ frames) during 100% occlusion.
    - **Metric:** Zero-drift latent retention confirmed via `verify_task17.py`.

- [x] **Task 18:** Edge Case Hardening: Handling Vanishing Nodes (Occlusion) ✅
    - *Outcome:* Integrated "Blink Logic" in Isaac Sim; verified 10% occlusion resilience in HDF5 stream.

- [] **Task 19:** Latent Separation Validation: Silhouette Score Benchmarking.

- [] **Task 20:** Phase 2 Review: GNN Performance Metrics & Inference Latency Audit.
---
*Note: This roadmap is a living document. We will check off tasks as we complete them.*