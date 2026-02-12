# 🧠 Neuro-Symbolic Multimodal System Design

## 1. System Philosophy
The **E-VLAformer** architecture resolves **Causal Hallucinations** in robot manipulation by coupling a high-capacity Transformer (Connectionist) with a Graph World Model (Symbolic). This hybrid approach ensures that the model's actions are grounded in physical reality rather than statistical patterns alone.

---

## 2. Graph World Model (GWM)
The GWM is the symbolic heart of the system, representing the environment as a structured graph $G = \{V, E\}$.

### 2.1 Node Representation ($V$)
Following the implementation in `src/models/graph_dataset.py`, nodes represent physical entities (joints, links, objects):
- **State Features:**
$$\mathbf{x}_i = [p_x, p_y, p_z, m, \text{type\_id} \in \mathbb{Z}]$$
- **Semantic Anchors:** Each node is anchored to a specific `sim_path` (Isaac Sim) or `hw_id` (DIY Robot MG996R). The use of integer `type_id` ensures strict categorical grounding for the GNN.

### 2.2 Relational Edges ($E$)
Edges define the physics-based constraints of the workspace:
- **Kinematic Edges ($E_{kin}$):** Represent the rigid hierarchical structure of the DIY 4-DOF arm ($Base \rightarrow Joint \rightarrow Link \rightarrow Gripper$).
- **Contact Edges ($E_{con}$):** Dynamic edges instantiated when Euclidean distance $dist(v_{gripper}, v_{obj}) < \epsilon$. These edges trigger the Reasoning Module's momentum transfer logic.



---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a linear encoder for immediate feedback.
- **Slow Path (30Hz):** RGB-D Video / Mobile Camera feed processed via a Vision Transformer (ViT).

### 3.2 Cross-Attention Alignment
We utilize a **Q-Former** style query mechanism to align visual tokens with the GWM nodes. This ensures that the "AI thought" (latent space) is physically consistent with the "Physical reality" (pixels).

---

## 4. Causal Reasoning Module (CRM)
The CRM acts as a **Physics Consistency Filter**:
1. **Prediction:** The VLA proposes a raw action $A_{raw}$.
2. **Simulation:** The GNN predicts the next state $S_{t+1}$ based on the Graph topology.
3. **Correction:** If $S_{t+1}$ violates a physical constraint (e.g., self-collision or phantom movement), the action is modified to $A_{safe}$.

### 4.1 Temporal Persistence (Phase 3 Goal)
The CRM is being upgraded with a **Graph Memory Buffer** (Task 17) to ensure nodes persist during occlusion (The "Lid Test"), solving the "Out-of-Sight, Out-of-Mind" hallucination.



---

## 5. Technical Implementation (Current Progress)

- [x] **Data Format:** Object-Centric HDF5 (Task 11)
- [x] **Graph Layer:** Relational Graph Builder & Edge Logic (Task 12)
- [x] **Graph Neural Network:** Message passing blocks (Task 13)
      - *Specs:* 3-layer GraphSAGE, 64 hidden channels.
- [x] **Multimodal Fusion:** Cross-Attention alignment (Task 14)
- [x] **Latent Manifold Analysis:** t-SNE Topology Audit (Task 15)
- [x] **Contrastive Physics Grounding:** InfoNCE Causal Separation (Task 16)
      - *Status:* **Verified.** Achieved **2.6x manifold expansion** (150 -> 400 units).
      - *Outcome:* Successfully separated "Safe Reach" and "Collision" states in latent space.
      - *Artifact:* `models/weights/gnn_contrastive_beta.pth`
- [ ] **Global State Persistence:** Graph Memory Buffer (Task 17)
- [ ] **Policy Head:** Action-Modulated Policy (Task 20+)

---
*Generated: 2026-02-12 | E-VLAformer Research Lab*