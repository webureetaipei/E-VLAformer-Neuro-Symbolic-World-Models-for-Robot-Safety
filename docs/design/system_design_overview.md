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

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Object-centric representations including physical attributes.
    * **Edges ($E$):** Causal and spatial relationships (Contact, Occlusion).
* **Cognitive Persistence (Task 17 & 18 Verified) ✅:**
    * **Task 17:** Implemented a TTL-based **Graph Memory Buffer** for object permanence.
    * **Task 18:** Integrated **Blink Logic** in Isaac Sim 4.5.0 to generate "Hardened" HDF5 datasets.
    * **Outcome:** The system maintains 100% feature parity during 30+ frame occlusion events, relying on cached GWM nodes when visual tokens vanish ($P(\text{visibility}) = 0$).


### 2.2 Topological Certification & Latent Audit
* **Manifold Monitoring:** Uses **t-SNE** to project GNN embeddings. Verified **2.6x Manifold Expansion** post-Task 16.
* **Task 19 (Active) 🚀:** Implementing the **Silhouette Latent Audit** to quantify topological stability during "Blink" events, ensuring memory representations do not drift in the latent space.

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship via gRPC and low-latency Serial/WebSockets.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset

### 4.1 Data Pipeline
* **Occlusion-Aware Generation (Task 18 Verified) ✅:** * Successfully generated `task18_occlusion_test_001.h5` with stochastic 10% blink rates.
    * Verified pixel-perfect synchronization between visibility toggles and causal ground-truth flags.
* **Storage (HDF5):** GZIP-compressed chunks optimized for 95% GPU utilization during training.

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
| **GWM Latent** | **Silhouette Score** | **> 0.55** | 🚀 **Task 19 Active** |
| **GWM Latent** | **Manifold Expansion Ratio** | **> 2.0x** | ✅ **2.6x Verified** |
| **GWM Latent** | **Object Permanence** | **> 30 Frames** | ✅ **30+ Frames (Verified)** |
| **GWM Latent** | **Occlusion Resilience** | **> 90%** | ✅ **100% (Hardened Data)** |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*