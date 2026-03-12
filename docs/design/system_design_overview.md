# 🦾 E-VLAformer System Design Overview

**Status:** Active (Phase 3 - Policy Evaluation)  
**Author:** Tsung Lung Yang  
**Target:** NeurIPS 2026  
**Core Framework:** Neuro-Symbolic Vision-Language-Action (VLA)  
**Official Repository:** [🤗 Hugging Face: TsungLungYang/E-VLAformer-GWM-Dataset](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset)

---

## 1. Embedded Optimization: The "TinyEngine" Architecture
**Goal:** Deploy VLA logic on a distributed edge-controller architecture with deterministic timing.

### 1.1 Core Implementation Strategy
* **Hybrid Inference Pipeline:**
    * **Brain (PC):** Executes ViT and Graph Reasoning using the C++ **TinyEngine**.
    * **Nervous System (ESP32):** Performs real-time IK and 50Hz PWM generation for MG996R servos.
* **Zero-Malloc Runtime:** Pre-calculated tensor offsets within a Linear Memory Arena eliminate runtime fragmentation.
* **HAL (Hardware Abstraction Layer):** Certified for **"Iron Grip"** logic. Maps simulation's negative joint positions (`-0.01`) to physical PWM duty-cycle saturation to ensure high-torque object clamping.

---

## 2. Multimodal AI System: Neuro-Symbolic VLA

### 2.1 Model Architecture
* **Graph World Model (GWM):**
    * **Nodes ($V$):** Object-centric representations including physical attributes (mass, friction, ID).
    * **Edges ($E$):** Causal and spatial relationships (Kinematic Constraints, Dynamic Contacts).
* **Cognitive Persistence (Verified) ✅:**
    * **Object Permanence:** TTL-based circular buffer to maintain graph nodes during visual dropout.
    * **Identity Mapping:** Applied **Identity Collapse** training to ensure latent representations are identical for "Visible" and "Occluded" states.
    * **Outcome:** Verified via Task 30 "Blind Grasp" tests; zero topological drift across extended occlusion events.

### 2.2 Multimodal Sensor Fusion & Training (Task 21-30 Verified) ✅
* **Policy Fusion:** Deployment of a Residual MLP fusing GNN latents, Joint-space proprioception, and Language embeddings.
* **Sensor Grounding:** Implementation of a calibrated Proprioception Handler. Normalizes raw $\pm 90^\circ$ joint angles to the $[-1, 1]$ latent manifold.
* **Behavioral Cloning Pipeline (Task 30):** **[ELITE STATUS]** * **Optimization:** Achieved full convergence via Huber Loss and `ReduceLROnPlateau` scheduling.
    * **Unmasked Gradient Flow:** Successfully "unmasked" GWM nodes during training, allowing the policy to learn direct spatial-to-action mapping.
    * **Elite Checkpoint:** The final policy is locked in **`evla_advanced_epoch80.pth`** with a **Huber Loss of 0.249**.
* **Robustness Production (Task 29) ✅:** Scaled the harvesting pipeline to generate a **100-episode "Robustness Trinity" dataset** hosted on the [Hugging Face Repository](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset).

---

## 3. Distributed System: Sim-to-Real Infrastructure
**Goal:** Maintain a "Digital Twin" relationship via gRPC (Protobuf) and low-latency Serial communication.

---

## 4. Data Engine Strategy: The "Audit-Ready" Dataset

### 4.1 Multi-Scenario Harvesting (Task 28-30 Verified) ✅
The E-VLAformer dataset is "Scenario-Aware," capturing both success and recovery data.
* **Master Weights:** **`evla_advanced_epoch80.pth`** (Certified for 4-DOF inference).
* **Scenario A-C:** Spatial Randomization (Normal, Left, Right offsets).
* **Scenario D: 阻擋 (Obstacle):** Robot navigates around cuboids while maintaining end-effector targets.
* **Scenario E: 碰撞 (Collision):** Captures "Out-of-Distribution" (OOD) joint vibrations and recovery deltas.
* **Robustness Trinity (Verified):**
    * **Full Occlusion (40%):** Trains GWM persistence via total visual deprivation.
    * **Dynamic Perturbation (27%):** Mid-trajectory target shifting to train reactive path re-planning.

---

## 5. Success Criteria (Engineering KPIs)

### 5.1 System Performance KPIs
| Component | Metric | Target | Current |
| :--- | :--- | :--- | :--- |
| **TinyEngine** | E2E Latency (PC+ESP32) | **< 20ms** | **14.2ms** |
| **Grip Stability** | Transport Success | **> 98%** | ✅ **100% (Iron Grip)** |
| **Policy Convergence** | **Final Huber Loss** | **< 0.30** | ✅ **0.249 (Epoch 80)** |

### 5.2 Latent Topology & Cognition KPIs
| Component | Metric | Target | Status |
| :--- | :--- | :--- | :--- |
| **GWM Latent** | **Silhouette Stability** | **$\approx 0.00$** | ✅ **0.0000 (Identity)** |
| **Official Weights** | **Technical Freeze** | **`evla_advanced_epoch80.pth`** | ✅ **Locked** |
| **Unified Fusion** | **Input Vector Dim** | **548-dim** | ✅ **Verified (Task 24)** |
| **Data Volume** | **Audited Episodes** | **100 Episodes** | ✅ **Certified (Task 29)** |

---
*Note: This document is a living blueprint for the E-VLAformer research initiative.*