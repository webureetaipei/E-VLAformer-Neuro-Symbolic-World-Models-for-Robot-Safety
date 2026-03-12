# 🧠 Neuro-Symbolic Multimodal System Design

## 1. System Philosophy
The **E-VLAformer** architecture resolves **Causal Hallucinations** in robot manipulation by coupling a high-capacity Transformer (Connectionist) with a Graph World Model (Symbolic). This hybrid approach ensures that the model's actions are grounded in physical reality rather than statistical patterns alone.

---

## 2. Graph World Model (GWM)
The GWM is the symbolic heart of the system, representing the environment as a structured graph $G = \{V, E\}$.

### 2.1 Node Representation ($V$)
Following the implementation in `src/models/graph_dataset.py`, nodes represent physical entities (joints, links, objects):
- **State Features:**
  $$\mathbf{x}_i = [p_x, p_y, p_z, m, \mathrm{type\_id} \in \mathbb{Z}]$$
- **Semantic Anchors:** Each node is anchored to a specific `sim_path` (Isaac Sim) or `hw_id`.

### 2.2 Relational Edges ($E$)
- **Kinematic Edges ($E_{kin}$):** Represent the rigid hierarchical structure of the Franka Emika Panda.
- **Contact Edges ($E_{con}$):** Dynamic edges instantiated when Euclidean distance $dist(v_{gripper}, v_{obj}) < \epsilon$.

---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a calibrated normalization handler.
- **Slow Path (30Hz):** RGB-D Video feed processed via a Vision Transformer (ViT).

---

## 4. Causal Reasoning Module (CRM)

### 4.1 Cognitive Persistence & Object Permanence ✅
To solve the "Out-of-Sight, Out-of-Mind" hallucination problem, we utilize a **Graph Memory Buffer** with **Identity Mapping**.
- **Object Permanence:** Verified via Task 30 "Blind Grasp" tests; the model maintains target-centric trajectories during 100% visual blackout.
- **Latent Identity Mapping (Task 19):** Manifold distance between "Visible" and "Occluded" states is mathematically zero ($S = 0.00$).

### 4.2 Elite Action Policy (Task 30 Verified) ✅
The action loop is grounded in real-time physical feedback and high-entropy stochastic demonstrations.
- **Elite Checkpoint:** All policy weights are frozen in **`evla_advanced_epoch80.pth`** ([Download via Hugging Face](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset)).
- **Optimization:** Achieved a **0.249 Huber Loss** through unmasked GWM gradient flow, enabling reactive path re-planning.
- **Robustness Trinity (Task 29):** 100 expert trajectories hosted on [Hugging Face](https://huggingface.co/datasets/TsungLungYang/E-VLAformer-GWM-Dataset).

| Normal Baseline | Visual Occlusion | Dynamic Perturbation |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/686e73ee-8ed3-43a1-b7f0-130a672a03f4" width="100%" controls></video> | <video src="https://github.com/user-attachments/assets/d2fec678-bcca-4b91-9817-e158e402f248" width="100%" controls></video> | <video src="https://github.com/user-attachments/assets/8eb4106a-51de-4b2c-aaae-7855b84c7b46" width="100%" controls></video> |

---

## 5. Technical Implementation (Progress Tracking)

- [x] **Global State Persistence:** Graph Memory Buffer (Task 17) ✅
- [x] **Silhouette Stability Audit:** Verified Identity Mapping $S = 0.00$ (Task 19) ✅
- [x] **Behavioral Cloning:** Expert Trajectory Trainer (Task 25) ✅ 
- [x] **Advanced Manipulation:** Multi-Phase Pick-and-Place & Iron Grip (Task 28) ✅
- [x] **Robustness Production:** 100-Episode "Robustness Trinity" Dataset (Task 29) ✅
- [x] **Elite Policy Convergence:** 80-Epoch Optimization & **`evla_advanced_epoch80.pth`** (Task 30) ✅

---
*Generated: 2026-03-12 | E-VLAformer Research Lab*