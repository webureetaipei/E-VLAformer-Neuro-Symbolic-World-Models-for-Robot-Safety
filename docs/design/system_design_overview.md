# 🦾 E-VLAformer System Design Overview

**Status:** Active (Phase 3 - Policy Integration)  
**Author:** Tsung Lung Yang  
**Target:** NeurIPS 2026  
**Core Framework:** Neuro-Symbolic Vision-Language-Action (VLA)  

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy VLA logic on a distributed edge-controller architecture with deterministic timing.

### 1.1 Core Implementation Strategy
* **Hybrid Inference Pipeline:**
    * **Brain (PC):** Executes ViT and Graph Reasoning using the C++ **TinyEngine**.
    * **Nervous System (ESP32):** Performs real-time IK and 50Hz PWM generation for MG996R servos.
* **Zero-Malloc Runtime:** Pre-calculated tensor offsets within a Linear Memory Arena eliminate runtime fragmentation.
* **HAL:** A unified C++ Hardware Abstraction Layer for both **USD-based joints** (Simulation) and **physical servos** (Hardware).

---

## 2. Multimodal AI System: Neuro-Symbolic VLA

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Object-centric representations including physical attributes (mass, friction, ID).
    * **Edges ($E$):** Causal and spatial relationships (Kinematic Constraints, Dynamic Contacts).
* **Cognitive Persistence (Task 17-20 Verified) ✅:**
    * **Object Permanence:** TTL-based circular buffer to maintain graph nodes during visual dropout.
    * **Identity Mapping (Task 19):** Applied **Identity Collapse** training to ensure latent representations are identical for "Visible" and "Occluded" states.
    * **Outcome:** Zero topological drift across 30+ frame occlusion events.

### 2.2 Multimodal Sensor Fusion & Training (Task 21-25 Verified) ✅
* **Policy Fusion (Task 21):** Deployment of a Residual MLP fusing GNN latents, Joint-space proprioception, and Language embeddings.
* **Sensor Grounding (Task 22):** Implementation of a calibrated Proprioception Handler. Normalizes raw $\pm 90^\circ$ joint angles to the $[-1, 1]$ latent manifold with integrated **Alpha-Filter smoothing** ($\alpha=0.7$).
* **Language Grounding (Task 23):** Integration of the **Language Handler**. Utilizes `all-distilroberta-v1` with a custom **768→512 Projection Layer**.
* **Live Inference Engine (Task 24):** Synchronized all asynchronous streams into a deterministic **548-dim fusion vector**. Verified that real-time "Pixels-to-Actions" pass-through is operational.
* **Behavioral Cloning Pipeline (Task 25):** Certified the `BCTrainer` gradient path. This enables supervised optimization of the policy head by mapping the 548-dim fusion vectors to expert joint deltas ($\Delta \theta$) with stable loss convergence.


### 2.3 Topological Certification & Latent Audit
* **Manifold Monitoring:** Utilizes **t-SNE** to project GNN embeddings. Verified a **2.6x Manifold Expansion** post-Task 16.
* **Stability Audit (Task 19/20) ✅:** Silhouette Audit confirmed that "Memory Nodes" are topologically indistinguishable from "Sensory Nodes" ($S = 0.00$).

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship via gRPC (Protobuf) and low-latency Serial communication.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset

### 4.1 Data Pipeline
* **Occlusion-Aware Generation (Task 18 Verified) ✅:** Generated `task18_occlusion_test_001.h5` with stochastic 10% blink rates and synchronized causal ground-truth.
* **Certification (Task 20) ✅:** Passed structural and entropy audits, ensuring the Phase 3 training set is artifact-free.

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
| **GWM Latent** | **Silhouette Stability** | **$\approx 0.00$** | ✅ **0.0000 (Identity)** |
| **GWM Latent** | **Embedding Variance** | **$> 0.10$** | ✅ **0.5309 (Rich)** |
| **Unified Fusion** | **Input Vector Dim** | **548-dim** | ✅ **Verified (Task 24)** |
| **BC Pipeline** | **Gradient Path** | **Certified** | ✅ **Verified (Task 25)** |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*