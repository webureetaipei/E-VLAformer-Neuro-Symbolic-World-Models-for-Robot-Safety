# 🦾 E-VLAformer System Design Overview

**Status:** Active (Phase 3 - Cognitive Persistence)  
**Author:** Tsung Lung Yang  
**Target:** NeurIPS 2026  
**Core Framework:** Neuro-Symbolic Vision-Language-Action (VLA)  

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy VLA logic on a distributed edge-controller architecture with deterministic timing.

### 1.1 Core Implementation Strategy
* **Hybrid Inference Pipeline:**
    * **Brain (PC):** Executes ViT and Graph Reasoning using the C++ **TinyEngine**.
    * **Nervous System (ESP32):** Performs real-time IK and 50Hz PWM generation.
* **Zero-Malloc Runtime:** Pre-calculated tensor offsets eliminate runtime fragmentation.
* **HAL:** A unified C++ interface for both **USD-based joints** (Sim) and **physical servos** (Hardware).

---

## 2. Multimodal AI System: Neuro-Symbolic VLA
**Goal:** Solve "Causal Hallucination" and "Temporal Reasoning Deficits."

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Represent environmental objects and the Robot's Segments. Stores physical attributes (position, mass).
    * **Edges ($E$):** Define causal relationships (Contacting, Obstructing). 
* **Cognitive Persistence (Task 17 Verified) 🚀:**
    * Implements a **Graph Memory Buffer** with a TTL (Time-To-Live) mechanism.
    * **Object Permanence:** If an object is occluded, the GWM retains the node features for **30+ frames**, allowing the robot to interact with objects it can no longer see.


### 2.2 Topological Certification & Latent Audit
* **Manifold Monitoring:** Uses **t-SNE** to project GNN embeddings. Verified **2.6x Manifold Expansion** post-Task 16.
* **Temporal Stability:** Task 17 ensures that "remembered" nodes do not drift in the latent space during occlusion, maintaining geometric consistency.

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship via gRPC and low-latency Serial/WebSockets.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset
**Goal:** Automatically generate and certify high-fidelity datasets.

### 4.1 Data Pipeline
* **Occlusion-Aware Generation (Task 18 Active):** Simulating "Blink Events" in Isaac Sim to train the model to rely on its Memory Buffer during sensor failure.
* **Storage (HDF5):** Uncompressed chunks for maximum GPU utilization (95%).

---

## 5. Success Criteria (Engineering KPIs)

### 5.1 System Performance KPIs
| Component | Metric | Target | Current |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | E2E Latency (PC+ESP32) | **< 20ms** | **14.2ms** |
| **TinyEngine** | Memory Footprint (RAM) | **< 500MB** | **412MB** |
| **Control Loop** | Consistency | **50Hz** | **50Hz (Fixed)** |

### 5.2 Latent Topology & Cognition KPIs
| Component | Metric | Target | Status |
| :--- | :--- | :--- | :--- |
| **GWM Latent** | **Silhouette Score** | **> 0.55** | 🟡 **0.42** |
| **GWM Latent** | **Manifold Expansion Ratio** | **> 2.0x** | ✅ **2.6x Verified** |
| **GWM Latent** | **Object Permanence** | **> 30 Frames** | ✅ **30+ Frames (Task 17)** |
| **GWM Latent** | **Occlusion Resilience** | **> 90%** | 🔵 **Task 18 Goal** |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*