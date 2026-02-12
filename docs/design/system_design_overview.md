# 🦾 E-VLAformer System Design Overview

**Status:** Active (Phase 2/3 - Brain-Body Integration)  
**Author:** Tsung Lung Yang  
**Target:** NeurIPS 2026  
**Core Framework:** Neuro-Symbolic Vision-Language-Action (VLA)  

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy VLA logic on a distributed edge-controller architecture with deterministic timing and minimal overhead.

### 1.1 Core Implementation Strategy
* **Hybrid Inference Pipeline:**
    * **Brain (PC/Local Server):** Executes heavy Vision-Transformer (ViT) and Graph Reasoning loops using a custom C++ **TinyEngine** to manage tensor operations without standard runtime bloat.
    * **Nervous System (ESP32):** A lightweight controller receiving "Action Tokens" via high-speed Serial. It performs real-time **Inverse Kinematics (IK)** and generates 50Hz PWM signals for MG996R servos.
* **Zero-Malloc Runtime (Static Arena):** Implements a **Linear Memory Arena** where all tensor memory offsets are pre-calculated during compilation. This eliminates runtime fragmentation and guarantees a stable memory footprint on embedded chips.
* **Real-Time Hardware Abstraction Layer (HAL):** A unified C++ interface that abstracts hardware. The high-level policy interacts with a "Joint Object," regardless of whether it is a **USD-based joint** in Isaac Sim or a **physical servo** on the DIY arm.

### 1.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **The Reality-Control Gap.** Eliminates the 100ms+ latency of Python-based serial communication by moving IK and PWM timing to the ESP32 firmware.
* **Trade-off:** **Flexibility vs. Determinism.** *Decision:* We sacrifice dynamic model branching to ensure the control loop never misses a 20ms window (50Hz), which is critical for physical robot stability.

---

## 2. Multimodal AI System: Neuro-Symbolic VLA
**Goal:** Solve "Causal Hallucination" and "Temporal Reasoning Deficits" in robot manipulation.

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Represent environmental objects and the **Robot's Segments** (Base, Link1, Link2, Gripper). Each node stores physical attributes like position and estimated mass.
    * **Edges ($E$):** Define causal relationships (e.g., *Contacting*, *Obstructing*). 
    * **Reasoning:** A lightweight GNN predicts future states. If the VLA's predicted action violates a graph constraint (e.g., "Moving through a solid wall"), the action is corrected via a symbolic safety layer.
* **Noise-Robust Modality Fusion:**
    * Uses **Cross-Attention** to align RGB-D video (30Hz), Proprioception (100Hz), and Language. 
    * Integrates a **Temporal Smoothing** layer to handle visual noise inherent in low-cost webcam inputs.



### 2.2 Topological Certification & Latent Audit
* **Metric-Driven World Modeling:** We use **t-SNE** to project 32-dimensional GNN physical embeddings into a 2D manifold. This serves as a "Structural Health Check" for the robot's world-state reasoning.
* **Geometric Consistency:** Verified that similar physical states cluster appropriately. This prevents "state-space aliasing" where the model confuses safe and dangerous configurations.
* **Verification Utility:** Automated audit scripts generate a Latent Manifold Report for every batch, ensuring that data-engine drift is caught before training.

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship between Isaac Sim and the DIY prototype.

### 3.1 Distributed Architecture
* **Decoupled Brain-Body Design:** Headless NVIDIA Isaac Sim instances running in Docker for massive data generation, synced with the physical ESP32-driven DIY Arm.
* **Communication Protocol:**
    * **gRPC (Protobuf):** High-throughput data transfer between sim nodes and the training cluster.
    * **Serial/WebSockets:** Low-latency feedback loop between the Inference Brain (PC) and Physical Arm (ESP32).

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset
**Goal:** Automatically generate and certify high-fidelity datasets that capture physical edge cases.

### 4.1 Data Pipeline
* **Sim-to-Real Feedback Loop:** Failure cases observed on the physical arm (e.g., servo stall) are tagged and used to generate targeted synthetic data in Isaac Sim for **Curriculum Learning**.
* **Storage (HDF5):** Tensors are stored in uncompressed **HDF5 chunks** to enable sequential disk reads, increasing GPU utilization from 60% to 95% by removing CPU decoding bottlenecks.

---

## 5. Success Criteria (Engineering KPIs)

### 5.1 System Performance KPIs
| Component | Metric | Target | Baseline (Implicit VLA) |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | E2E Latency (PC+ESP32) | **< 20ms** | ~100ms |
| **TinyEngine** | Memory Footprint (RAM) | **< 500MB** | > 2.5GB |
| **Control Loop** | Consistency | **50Hz** | 20Hz (Variable) |
| **Sim-to-Real** | Pose Error (DIY Arm) | **< 8mm** | N/A |

### 5.2 Latent Topology KPIs
| Component | Metric | Target | Status |
| :--- | :--- | :--- | :--- |
| **GWM Latent** | **Silhouette Score** | **> 0.55** | 🟡 **0.42** (Post-Task 16 Improvement) |
| **GWM Latent** | **Manifold Expansion Ratio** | **> 2.0x** | ✅ **2.6x** (Task 16 Verified) |
| **GWM Latent** | **Collision Separation** | **$d(Safe, Coll) > \sigma$** | 🔵 Phase 3 Goal |
| **GWM Latent** | **Temporal Continuity** | **$\Delta L < \epsilon$** | 🔵 Task 17 Goal |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*