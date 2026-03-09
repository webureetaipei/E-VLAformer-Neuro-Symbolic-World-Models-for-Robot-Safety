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
- **Kinematic Edges ($E_{kin}$):** Represent the rigid hierarchical structure of the DIY 4-DOF arm ($Base \rightarrow Joint \rightarrow Link \rightarrow Gripper$).
- **Contact Edges ($E_{con}$):** Dynamic edges instantiated when Euclidean distance $dist(v_{gripper}, v_{obj}) < \epsilon$.

---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a calibrated normalization handler.
- **Slow Path (30Hz):** RGB-D Video / Mobile Camera feed processed via a Vision Transformer (ViT).

---

## 4. Causal Reasoning Module (CRM)

### 4.1 Cognitive Persistence & Object Permanence ✅
To solve the "Out-of-Sight, Out-of-Mind" hallucination problem, we utilize a **Graph Memory Buffer** with **Identity Mapping**.
- **The Lid Test (Task 18):** Verified that the GWM retains node attributes even when visibility $P=0$ due to physical occlusion.
- **Latent Identity Mapping (Task 19):** Manifold distance between "Seen" and "Remembered" objects is mathematically zero ($S = 0.00$).

### 4.2 Action Loop & Robustness (Task 21-29 Verified) ✅
The action loop is grounded in real-time physical feedback and high-entropy stochastic demonstrations:
- **Iron Grip Protocol (Task 28):** Forces negative joint commanding ($[-0.01, -0.01]$) to maintain torque saturation.
- **Robustness Trinity (Task 29):** Mass-produced 50 expert trajectories covering Baseline, Occlusion, and Perturbation scenarios.

| Normal Baseline | Visual Occlusion | Dynamic Perturbation |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/51a7333a-c138-4abe-b1b0-2d84bc367c40" width="100%"></video> | <video src="https://github.com/user-attachments/assets/0be5fa49-3628-478a-9c24-0fe5bb534e3d" width="100%"></video> | <video src="https://github.com/user-attachments/assets/b4f42cc8-ef76-4212-956c-a51b0f0ff03e" width="100%"></video> |

---

## 5. Technical Implementation (Progress Tracking)

- [x] **Global State Persistence:** Graph Memory Buffer (Task 17) ✅
- [x] **Silhouette Stability Audit:** Verified Identity Mapping $S = 0.00$ (Task 19) ✅
- [x] **Behavioral Cloning:** Expert Trajectory Trainer (Task 25) ✅ 
- [x] **Advanced Manipulation:** Multi-Phase Pick-and-Place & Iron Grip (Task 28) ✅
- [x] **Robustness Production:** 50-Episode "Robustness Trinity" Dataset (Task 29) ✅
- [ ] **Unified Training Pipeline:** Master HDF5 DataLoader & BC Optimization (Task 30)

---
*Generated: 2026-03-09 | E-VLAformer Research Lab*