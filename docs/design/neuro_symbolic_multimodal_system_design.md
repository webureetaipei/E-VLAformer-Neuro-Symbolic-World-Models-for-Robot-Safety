# Neuro-Symbolic Multimodal System Design

## 1. System Philosophy
The E-VLAformer architecture aims to resolve **Causal Hallucinations** in robot manipulation by coupling a high-capacity Transformer (Connectionist) with a Graph World Model (Symbolic). This hybrid approach ensures that the model's actions are grounded in physical reality rather than statistical patterns alone.

---

## 2. Graph World Model (GWM)
The GWM is the symbolic heart of the system, representing the environment as a structured graph $G = \{V, E\}$.

### 2.1 Node Representation ($V$)
Following the implementation in `src/models/graph_dataset.py` (Task 11), nodes represent physical entities:
- **State Features:**
$$\mathbf{x}_i = [p_x, p_y, p_z, m, \mathrm{type\_id}]$$
- **Semantic Anchors:** Each node is anchored to a specific `sim_path` (Isaac Sim) or `hw_id` (DIY Robot MG996R).

### 2.2 Relational Edges ($E$)
Edges are procedurally generated via the `RelationalGraphBuilder` (Task 12) to define physics-based constraints:

- **Kinematic Edges ($E_{kin}$):** Represent the rigid hierarchical structure of the DIY 4-DOF arm ($Base \rightarrow Joint \rightarrow Link \rightarrow Gripper$). These are static and bi-directional to allow gradient flow during backpropagation.
- **Contact Edges ($E_{con}$):** Dynamic edges instantiated when Euclidean distance $dist(v_{gripper}, v_{obj}) < \epsilon$. These edges trigger the CRM's momentum transfer logic.

### 2.3 Implementation Progress: From Tensors to Logic
We have successfully implemented the "Bridge" between raw sensory data and symbolic reasoning. The system now creates a structured mapping between the Physical World (Sensors) and the AI Brain (Graph States).

---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a linear encoder for immediate feedback.
- **Slow Path (30Hz):** RGB-D Video / Mobile Camera feed processed via a Vision Transformer (ViT).

### 3.2 Cross-Attention Alignment
We utilize a **Q-Former** style query mechanism to align visual tokens with the GWM nodes, ensuring that the "AI thought" matches the "Physical reality".

---

## 4. Causal Reasoning Module (CRM)
The CRM acts as a **Physics Consistency Filter**.

1. **Prediction:** The VLA proposes an action $A_{raw}$.
2. **Simulation:** The GNN predicts the next state $S_{t+1}$ based on the Graph.
3. **Correction:** If $S_{t+1}$ violates a physical constraint (e.g., self-collision), the action is modified to $A_{safe}$.

---

## 5. Technical Implementation (Current Progress)
- [x] **Data Format:** Object-Centric HDF5 (Task 11)
- [x] **Graph Layer:** Relational Graph Builder & Edge Logic (Task 12)
- [x] **Graph Neural Network:** Message passing blocks (Task 13)
- [ ] **Policy Head:** Action-Modulated Policy (Task 20+)