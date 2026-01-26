# E-VLAformer System Design Overview

**Status:** Draft (Phase 1)
**Author:** Tsung Lung Yang
**Target:** NeurIPS 2026

This document outlines the architectural decisions, trade-offs, and optimization strategies for the E-VLAformer project. It serves as the master blueprint (High-Level Design) for our **Neuro-Symbolic VLA** implementation. Detailed Low-Level Designs (LLD) for each subsystem will follow in separate documents as referenced below.

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy large VLA models on edge devices (e.g., Jetson Orin) with **<10ms latency** and **Zero-Dependency**.

### 1.1 Core Implementation Strategy
We reject standard runtimes (PyTorch/ONNX) in favor of a bare-metal C++ approach.

* **Zero-Malloc Runtime (Static Arena):**
    * *Mechanism:* We implement a **Linear Memory Arena**. Instead of `malloc/free`, all tensor memory offsets are pre-calculated during the compilation phase using a **Liveness Analysis Algorithm**.
    * *Benefit:* Eliminates runtime overhead and memory fragmentation. Guarantees 100% stable memory footprint.
* **Paged KV-Cache Manager (VLA Specific):** 
    * *Mechanism:* To handle autoregressive generation without memory fragmentation, we implement a **PagedAttention-style** block manager. Key-Value states are stored in non-contiguous pre-allocated memory blocks.
    * *Benefit:* Enables long-context reasoning for robot task planning without OOM (Out-Of-Memory) crashes.
* **Int8 Post-Training Quantization (PTQ):**
    * *Mechanism:* We utilize **Symmetric Per-Channel Quantization** for weights and activations. The engine implements custom `Int8 GEMM` kernels optimized for ARM NEON instructions.
    * *Benefit:* Reduces VRAM usage by 4x and increases throughput by ~3x on Jetson Orin (DLLA).
* **AOT (Ahead-of-Time) Code Generation:**
    * *Mechanism:* Python model definitions are transpiled into standard C++17 source code (no external libraries).
    * *Optimization:* We utilize **Loop Unrolling** and **SIMD (NEON)** instructions explicitly generated for ARMv8 architectures.
* **Operator Fusion:**
    * We aggressively fuse `Conv2d + BatchNorm + ReLU` and `Linear + Gelu` layers to minimize VRAM Global Memory Access (the primary bottleneck in Transformer inference).

### 1.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **The Memory Wall**. Python runtimes typically incur ~300MB overhead; our engine targets **<5MB** system overhead.
* **Trade-off:** **Flexibility vs. Performance**.
    * *Decision:* We sacrifice **Dynamic Computation Graphs** (no control flow inside the model) to achieve **Deterministic Latency**.
    * *Reference:* See `docs/design/004_tiny_vla_engine.md` (Planned) for memory offset algorithms.

---

## 2. Multimodal AI System: Neuro-Symbolic VLA
**Goal:** Solve the "Temporal Reasoning Deficit" and "Causal Hallucination" in current Vision-Language-Action models.

### 2.1 Model Architecture
We propose a hybrid Neuro-Symbolic architecture merging Connectionist (Deep Learning) and Symbolic (Graph) methods.

* **Graph World Model (GWM):**
    * *Mechanism:* A dynamic **Scene Graph** ($G = \{V, E\}$) serves as the world state. Visual patches are mapped to nodes $V$ (Objects), and temporal changes update edges $E$ (Relations).
    * *Reasoning:* We use a lightweight **Graph Neural Network (GNN)** to predict future states based on causal rules, acting as a "Physics Consistency Filter" for the VLA output.
* **Hierarchical Modality Fusion:**
    * **Fast Path (100Hz):** Proprioception (Joint angles/Torque) $\rightarrow$ MLP Encoder $\rightarrow$ Action Head.
    * **Slow Path (30Hz):** RGB-D Video $\rightarrow$ Vision Transformer (ViT) $\rightarrow$ Cross-Attention.
    * **Alignment:** Utilizes **Q-Former** style queries to align asynchronous modalities.

### 2.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **Safety via Causality**. Standard LLMs "guess" physics probabilistically. Our Graph Module enforces logical constraints (e.g., *Constraint: "Grip" requires "Contact"*).
* **Trade-off:** **Training Complexity vs. Inference Safety**.
    * *Decision:* We accept a complex multi-stage training pipeline (Pre-training $\rightarrow$ Graph Alignment $\rightarrow$ Action Finetuning) to ensure **Safety-Critical** behavior.
    * *Reference:* See `docs/design/002_graph_reasoning.md` (Planned) for GNN formulations.

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Scale data generation to **1,000+ hours** of synthetic robot interaction using heterogeneous compute clusters.

### 3.1 Distributed Architecture
* **Decoupled "Brain-Body" Design:**
    * **The Body (Simulation):** Headless NVIDIA Isaac Sim instances running in Docker containers.
    * **The Brain (Inference):** Policy networks running on separate GPU clusters.
* **Communication Protocol:**
    * Utilizes **gRPC (HTTP/2)** with **Protobuf** serialization for low-latency (<2ms) state transfer.
    * Implements **Asyncio** streaming to handle 100Hz control loops over the network without blocking.

### 3.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **Rendering/Physics Resource Contention**. By separating Physics Simulation (GPU PhysX) from Neural Inference (Tensor Cores), we maximize hardware utilization on separate machines.
* **Trade-off:** **Network Latency vs. Scalability**.
    * *Decision:* We introduce minor network latency (1-2ms) to gain **Infinite Scalability** (linear scaling of data generation nodes on Cloud/Kubernetes).
    * *Reference:* See `docs/design/005_scalability_ops.md` (Planned) for gRPC definitions.
    ---

## 4. Data Engine Strategy: The "Infinite" Dataset
**Goal:** Auto-generate high-quality synthetic data to overcome the "Real-World Data Scarcity" problem.

### 4.1 Data Pipeline
* **Generation (Sim):** Headless Isaac Sim nodes generate `RGB-D + Proprioception + Semantic Labels` at 100Hz.
* **Filtering (Auto-Labeling):** A rule-based "Sanity Check" filter discards failed episodes (e.g., robot self-collision) before saving to disk.
* **Storage:** Data is serialized into **HDF5 (Hierarchical Data Format)** chunks, optimized for high-throughput I/O during training.

### 4.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **I/O Blocking during Training**. Loading millions of small image files kills GPU utilization.
    * *Solution:* We aggregate data into large HDF5 chunks (2GB each) to enable sequential disk reads.
* **Trade-off:** **Storage Cost vs. Training Speed**.
    * *Decision:* We store uncompressed raw tensors (Float16) instead of JPG/PNG to avoid CPU decoding overhead during training, trading disk space for faster epoch times.

---

## 5. Success Criteria (Engineering KPIs)
We define the project's success through strict quantitative metrics.

### 5.1 System Performance KPIs
| Component | Metric | Target | Baseline (PyTorch) |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | End-to-End Latency | **< 10ms** | ~45ms |
| **TinyEngine** | Memory Footprint (RAM) | **< 500MB** | > 2.5GB |
| **Sim-to-Real** | Control Frequency | **100Hz** | 30Hz |
| **Distributed** | Data Gen Throughput | **100 Episodes/min** | 10 Episodes/min |

### 5.2 Research Success Metrics (NeurIPS)
* **Success Rate:** Achieve >90% success rate on "Long-Horizon Manipulation Tasks" (e.g., *Make coffee*).
* **Safety Violation Rate:** Reduce collision rate to **< 0.1%** using the Graph World Model (vs. 5% in standard End-to-End policies).