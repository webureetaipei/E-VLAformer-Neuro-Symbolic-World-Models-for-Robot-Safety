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
---
*Note: This roadmap is a living document. We will check off tasks as we complete them.*