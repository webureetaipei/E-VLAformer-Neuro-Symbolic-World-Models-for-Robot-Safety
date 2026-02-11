# E-VLAformer System Design Overview

**Status:** Status: Active (Phase 2/3 - Brain-Body Integration)
**Author:** Tsung Lung Yang
**Target:** NeurIPS 2026

This document defines the architectural blueprint for the **E-VLAformer**, a Neuro-Symbolic Vision-Language-Action model. The system is engineered to bridge high-fidelity simulation (NVIDIA Isaac Sim) with low-cost physical deployment (DIY ESP32-based Robotic Arm) through a unified causal reasoning framework.

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy VLA logic on a distributed edge-controller architecture with deterministic timing and minimal overhead.

### 1.1 Core Implementation Strategy
* **Hybrid Inference Pipeline:**
    * **Brain (PC/Local Server):** Executes heavy Vision-Transformer and Graph Reasoning loops using a custom C++ **TinyEngine** to manage tensor operations without standard runtime bloat.
    * **Nervous System (ESP32):** A lightweight controller receiving "Action Tokens" via high-speed Serial (USB) or WebSocket. It performs real-time **Inverse Kinematics (IK)** and generates 50Hz PWM signals for the MG996R servos.
* **Zero-Malloc Runtime (Static Arena):** * Implements a **Linear Memory Arena** where all tensor memory offsets are pre-calculated during compilation. This eliminates runtime fragmentation and guarantees a stable memory footprint.
* **Real-Time Hardware Abstraction Layer (HAL):** * A unified C++ interface that abstracts hardware. The high-level policy interacts with a "Joint Object," regardless of whether it is a **USD-based joint** in Isaac Sim or a **physical servo** on the DIY arm.

### 1.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **The Reality-Control Gap.** Eliminates the 100ms+ latency of Python-based serial communication by moving IK and PWM timing to the ESP32 firmware.
* **Trade-off:** **Flexibility vs. Determinism.** * *Decision:* We sacrifice dynamic model branching (no conditional control flow inside the model) to ensure the control loop never misses a 20ms window (50Hz), which is critical for physical robot stability.

---

## 2. Multimodal AI System: Neuro-Symbolic VLA
**Goal:** Solve "Causal Hallucination" and "Temporal Reasoning Deficits" in robot manipulation.

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Represent environmental objects and the **Robot's Segments** (Base, Link1, Link2, Gripper). Each node stores physical attributes like position and estimated mass.
    * **Edges ($E$):** Define causal relationships (e.g., *Contacting*, *Obstructing*). 
    * **Reasoning:** A lightweight GNN predicts future states. If the VLA's predicted action violates a graph constraint (e.g., "Moving through a solid wall"), the action is corrected.
* **Noise-Robust Modality Fusion:**
    * Uses **Cross-Attention** to align RGB-D video (30Hz), Proprioception (100Hz), and Language. 
    * Integrates a **Temporal Smoothing** layer to handle visual noise inherent in low-cost webcam inputs.

### 2.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **Safety in Unseen Scenarios.** Unlike end-to-end models that "guess" safety, the GWM provides a hard symbolic check against physical laws.
* **Trade-off:** **Model Size vs. Reasoning Depth.** * *Decision:* We utilize a shallow GNN (3-layer) to keep inference under 10ms, sacrificing complex multi-object chain reasoning for immediate reaction speed.

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship between Isaac Sim and the DIY prototype.

### 3.1 Distributed Architecture
* **Decoupled Brain-Body Design:**
    * **Simulation Body:** Headless NVIDIA Isaac Sim instances running in Docker for massive data generation.
    * **Physical Body:** ESP32-driven DIY Arm for real-world verification.
* **Communication Protocol:**
    * **gRPC (Protobuf):** High-throughput data transfer between sim nodes and the training cluster.
    * **Serial/WebSockets:** Low-latency local feedback loop between the Inference Brain (PC) and Physical Arm (ESP32).

### 3.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **Data Scarcity.** Overcomes the lack of physical training data by using the simulation to "pre-train" the causal graph before real-world deployment.
* **Trade-off:** **Synchronicity vs. Throughput.** * *Decision:* We use **Asynchronous gRPC** for data collection to maximize throughput, accepting minor temporal drift (1-2ms) to gain 10x faster dataset accumulation.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset
**Goal:** Automatically generate and certify high-fidelity datasets that capture physical edge cases.

### 4.1 Data Pipeline
* **Sim-to-Real Feedback Loop:** * Failure cases observed on the physical DIY arm (e.g., servo stall due to weight) are tagged and used to generate targeted synthetic data in Isaac Sim for **Curriculum Learning**.
* **Storage (HDF5):** * Tensors are stored in uncompressed **HDF5 chunks** (2GB/chunk) to enable sequential disk reads, preventing I/O bottlenecks during training.

### 4.2 Bottlenecks & Trade-offs
* **Bottleneck Solved:** **CPU-Bound Decoding.** By storing raw tensors instead of compressed PNGs, we remove the CPU decoding bottleneck, increasing GPU utilization from 60% to 95%.
* **Trade-off:** **Disk Space vs. Training Speed.** * *Decision:* We trade ~2TB of disk space (raw Float16 storage) to reduce total training time by 40%.

---

## 5. Success Criteria (Engineering KPIs)

### 5.1 System Performance KPIs
| Component | Metric | Target | Baseline (Implicit VLA) |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | E2E Latency (PC+ESP32) | **< 20ms** | ~100ms |
| **TinyEngine** | Memory Footprint (RAM) | **< 500MB** | > 2.5GB |
| **Control Loop** | Consistency | **50Hz** | 20Hz (Variable) |
| **Sim-to-Real** | Pose Error (DIY Arm) | **< 8mm** | N/A |

### 5.2 Research Success Metrics (NeurIPS)
* **Zero-Shot Transfer:** The model trained in Isaac Sim must execute "Pick-and-Place" on the DIY arm without real-world fine-tuning.
* **Causal Robustness:** Reduce collision rates to **< 0.1%** using the Graph World Model in scenarios with moving obstacles.