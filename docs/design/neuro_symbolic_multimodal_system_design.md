# Neuro-Symbolic Multimodal System Design

## 1. System Philosophy
The E-VLAformer architecture aims to resolve **Causal Hallucinations** in robot manipulation by coupling a high-capacity Transformer (Connectionist) with a Graph World Model (Symbolic).

---

## 2. Graph World Model (GWM)
The GWM is the symbolic heart of the system, representing the environment as a structured graph $G = \{V, E\}$.

### 2.1 Node Representation ($V$)
Following the implementation in `src/models/graph_dataset.py` (Task 11), nodes represent physical entities:
- **State Features:** $\mathbf{x}_i = [p_x, p_y, p_z, m, \text{type\_id}]$
- **Semantic Anchors:** Each node is anchored to a specific `sim_path` (Isaac Sim) or `hw_id` (DIY Robot).

### 2.2 Relational Edges ($E$)
Edges define physics-based constraints and spatial relations:
- **Kinematic Edges:** Define permanent links between robot segments (Base -> Joint -> Link).
- **Contact Edges:** Dynamic edges formed when $dist(v_i, v_j) < \epsilon$, triggering momentum transfer logic.

### 2.3 Implementation Progress: From Tensors to Logic
We have successfully implemented the "Bridge" between raw sensory data and symbolic reasoning.

**Visualizing the Achievement:** The system now creates a structured mapping between the Physical World (Sensors) and the AI Brain (Graph States). 

**Next Milestone: Task 12 - Relational Graph Construction**
We are transitioning from static data loading to dynamic rule definition. In Task 12, we will define the "Physics Rules"—mathematically linking Joint A movements to End-Effector B trajectories via hierarchical constraints.

---

## 3. Multimodal Fusion Engine
To handle the "Reality Gap" and asynchronous sensors, the system utilizes a Hierarchical Fusion strategy.

### 3.1 Asynchronous Streams
- **Fast Path (100Hz):** Proprioception (Joint angles) processed via a linear encoder for immediate feedback.
- **Slow Path (30Hz):** RGB-D Video / Mobile Camera feed processed via a Vision Transformer (ViT).

### 3.2 Cross-Attention Alignment
We utilize a **Q-Former** style query mechanism to align visual tokens with the GWM nodes, ensuring that the "AI thought" matches the "Physical reality."

---

## 4. Causal Reasoning Module (CRM)
The CRM acts as a **Physics Consistency Filter**.

1. **Prediction:** The VLA proposes an action $A_{raw}$.
2. **Simulation:** The GNN predicts the next state $S_{t+1}$ based on the Graph.
3. **Correction:** If $S_{t+1}$ violates a physical constraint (e.g., self-collision), the action is modified to $A_{safe}$.

---

## 5. Technical Implementation (Current Progress)
- [x] **Data Format:** Object-Centric HDF5 (Task 11)
- [ ] **Graph Layer:** GNN message passing blocks (Task 12)
- [ ] **Policy Head:** Action-Modulated Policy (Task 20+)