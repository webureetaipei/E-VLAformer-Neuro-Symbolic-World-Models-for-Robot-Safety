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
- **Semantic Anchors:** Each node is anchored to a specific `sim_path` (Isaac Sim) or `hw_id`. The use of integer `type_id` ensures strict categorical grounding for the GNN.

### 2.2 Relational Edges ($E$)
- **Kinematic Edges ($E_{kin}$):** Represent the rigid hierarchical structure of the DIY 4-DOF arm ($Base \rightarrow Joint \rightarrow Link \rightarrow Gripper$).
- **Contact Edges ($E_{con}$):** Dynamic edges instantiated when Euclidean distance $dist(v_{gripper}, v_{obj}) < \epsilon$. These edges trigger the Reasoning Module's momentum transfer logic.

---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a **calibrated normalization handler** for immediate feedback.
- **Slow Path (30Hz):** RGB-D Video / Mobile Camera feed processed via a Vision Transformer (ViT).

### 3.2 Cross-Attention Alignment
We utilize a **Q-Former** style query mechanism to align visual tokens with the GWM nodes. This ensures that the "AI thought" (latent space) is physically consistent with the "Physical reality" (pixels).

---

## 4. Causal Reasoning Module (CRM)
The CRM acts as a **Physics Consistency Filter** and **Cognitive Anchor**:

### 4.1 Cognitive Persistence (Task 17-20 Verified) ✅
To solve the "Out-of-Sight, Out-of-Mind" hallucination problem (occlusion), we implemented a **Graph Memory Buffer** with **Identity Mapping**.

- **Temporal Anchoring:** Nodes are assigned a $TTL_{max}$ (Time-To-Live) of 30 frames.
- **Latent Identity Mapping (Task 19):** We enforced **Identity Collapse** to ensure the manifold distance between a "Seen" object and a "Remembered" object is mathematically zero ($S = 0.00$).
- **Verification:** Variance analysis ($\sigma^2 = 0.53$) confirms that the model maintains a rich representation of the object even during 100% occlusion.

### 4.2 Proprioception & Action Loop (Tasks 21-27 Verified) ✅
The action loop is grounded in real-time physical feedback and high-entropy synthetic demonstrations:
- **Normalization (Task 22):** Raw joint angles ($\pm 90^\circ$) mapped to the $[-1, 1]$ unit range.
- **Language Grounding (Task 23):** Utilizes a **768 → 512 Projection Layer** to align semantic instructions.
- **Synchronized Inference (Task 24):** Orchestrates GNN, Proprioception, and Language into a unified **548-dim fusion vector**.
- **Data Harvesting (Task 26-27):** High-speed HDF5 engine with **Automated Movement Auditing** and **Domain Randomization** to ensure expert trajectories are verified and unfrozen.



### 4.3 Action Correction Loop
1. **Prediction:** The VLA proposes a raw action $A_{raw}$.
2. **Simulation:** The GNN predicts the next state $S_{t+1}$ based on the persistent Graph topology.
3. **Correction:** If $S_{t+1}$ violates a physical constraint (e.g., self-collision), the CRM modifies the output to $A_{safe}$.

---

## 5. Technical Implementation (Progress Tracking)

- [x] **Data Format:** Object-Centric HDF5 (Task 11)
- [x] **Graph Neural Network:** 3-layer GraphSAGE architecture (Task 13)
- [x] **Latent Manifold Analysis:** t-SNE Topology Audit (Task 15)
- [x] **Contrastive Physics Grounding:** InfoNCE Causal Separation (Task 16)
- [x] **Global State Persistence:** Graph Memory Buffer (Task 17) ✅
- [x] **Edge Case Hardening:** Occlusion Resilience / Blink Tests (Task 18) ✅
- [x] **Silhouette Stability Audit:** Verified Identity Mapping $S = 0.00$ (Task 19) ✅
- [x] **Phase 2 Technical Freeze:** Certified weights `certified_gwm_v1` (Task 20) ✅
- [x] **Policy Head:** VLA Action-Policy Architecture (Task 21) ✅ 
- [x] **Proprioception Handler:** Real-time Joint Normalization (Task 22) ✅ 
- [x] **Inference Engine:** Multimodal Sync & Live Control (Task 24) ✅ 
- [x] **Behavioral Cloning:** Expert Trajectory Trainer (Task 25) ✅ 
- [x] **Expert Data Engine:** High-Speed Harvester (Task 26) ✅
- [x] **Quality Audit:** Domain Randomization & Pixel MAD Test (Task 27) ✅
- [ ] **Advanced Manipulation:** Multi-Phase Pick-and-Place Logic (Task 28) 🚀 *ACTIVE*

---
*Generated: 2026-02-27 | E-VLAformer Research Lab*