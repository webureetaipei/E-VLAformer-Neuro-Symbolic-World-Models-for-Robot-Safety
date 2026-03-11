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
* **HAL (Hardware Abstraction Layer):** Certified for **"Iron Grip"** logic. Maps simulation's negative joint positions (`-0.01`) to physical PWM duty-cycle saturation to ensure high-torque object clamping on MG996R hardware.

---

## 2. Multimodal AI System: Neuro-Symbolic VLA

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Object-centric representations including physical attributes (mass, friction, ID).
    * **Edges ($E$):** Causal and spatial relationships (Kinematic Constraints, Dynamic Contacts).
* **Cognitive Persistence (Task 17-20 & 29 Verified) ✅:**
    * **Object Permanence:** TTL-based circular buffer to maintain graph nodes during visual dropout.
    * **Identity Mapping (Task 19):** Applied **Identity Collapse** training to ensure latent representations are identical for "Visible" and "Occluded" states.
    * **Outcome:** Verified via Task 29 "Blind Grasp" episodes; zero topological drift ($S=0.00$) across extended occlusion events.

### 2.2 Multimodal Sensor Fusion & Training (Task 21-29 Verified) ✅
* **Policy Fusion (Task 21):** Deployment of a Residual MLP fusing GNN latents, Joint-space proprioception, and Language embeddings.
* **Sensor Grounding (Task 22):** Implementation of a calibrated Proprioception Handler. Normalizes raw $\pm 90^\circ$ joint angles to the $[-1, 1]$ latent manifold with integrated **Alpha-Filter smoothing** ($\alpha=0.7$).
* **Language Grounding (Task 23):** Integration of the **Language Handler**. Utilizes `all-distilroberta-v1` with a custom **768→512 Projection Layer**.
* **Live Inference Engine (Task 24):** Synchronized all asynchronous streams into a deterministic **548-dim fusion vector**.
* **Behavioral Cloning Pipeline (Task 25):** Certified the `BCTrainer` gradient path. Enables supervised optimization of the policy head by mapping 548-dim vectors to expert joint deltas ($\Delta \theta$).
* **Data Harvesting Engine (Task 26) ✅:** Implementation of the high-speed HDF5 harvester. Fixed renderer synchronization to prevent frozen frames.
* **Domain Randomization (Task 27) ✅:** Verified environmental entropy (color/position) and integrated **Automated Movement Auditing** via Mean Absolute Difference (MAD) pixel analysis.
* **Advanced Manipulation (Task 28) ✅:** Implemented **Multi-Phase State Machines** for complex Pick-and-Place. Integrated **"Iron Grip" physics** to eliminate object slippage during transport.
* **Robustness Production (Task 29) ✅:** Scaled the harvesting pipeline to generate a **100-episode "Robustness Trinity" dataset**  [Hugging Face Datasets](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset). Certified stochastic scenario injection for **Visual Occlusion** and **Dynamic Perturbation** (Target Jumps).

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship via gRPC (Protobuf) and low-latency Serial communication.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset

### 4.1 Multi-Scenario Harvesting (Task 28-29 Verified) ✅
The E-VLAformer dataset is now "Scenario-Aware," capturing both success and recovery data. It is hosted and versioned for distributed training:
* **Dataset Repo:** [🤗 Hugging Face: E-VLAformer-GWM-Dataset](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset)
* **Scenario A-C:** Spatial Randomization (Normal, Left, Right offsets).
* **Scenario D: 阻擋 (Obstacle):** Robot uses RMPFlow to navigate around cuboids while maintaining end-effector targets.
* **Scenario E: 碰撞 (Collision):** Captures "Out-of-Distribution" (OOD) joint vibrations and recovery deltas.
* **Robustness Trinity (Task 29):**
    * **Full Occlusion (40%):** Trains GWM persistence via total visual deprivation (Giant Wall scenario).
    * **Dynamic Perturbation (27%):** Mid-trajectory target shifting to train reactive path re-planning.

---

## 5. Success Criteria (Engineering KPIs)

### 5.1 System Performance KPIs
| Component | Metric | Target | Current |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | E2E Latency (PC+ESP32) | **< 20ms** | **14.2ms** |
| **Grip Stability** | Transport Success | **> 98%** | ✅ **100% (Iron Grip)** |
| **Data Engine** | **Pixel Audit Status** | **Certified** | ✅ **MAD Verified (Task 27)** |

### 5.2 Latent Topology & Cognition KPIs
| Component | Metric | Target | Status |
| :--- | :--- | :--- | :--- |
| **GWM Latent** | **Silhouette Stability** | **$\approx 0.00$** | ✅ **0.0000 (Identity)** |
| **Expert Data** | **Scenario Coverage** | **3/3 Trinity** | ✅ **Certified (Task 29)** |
| **Unified Fusion** | **Input Vector Dim** | **548-dim** | ✅ **Verified (Task 24)** |
| **BC Pipeline** | **Gradient Path** | **Certified** | ✅ **Verified (Task 25)** |
| **Data Volume** | **Audited Episodes** | **100 Episodes** | ✅ **Mass-Produced (Task 29)** |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*