# 📊 E-VLAformer: Evaluation & Benchmarks

This report tracks the quantitative performance of the **E-VLAformer** architecture. It serves as the primary evidence for the NeurIPS 2026 submission, documenting the transition from raw simulation to a safety-aware Neuro-Symbolic world model.

---

## 1. World Model Quality: Latent Space Topology
We evaluate the Graph World Model (GWM) by analyzing its latent space. High-quality world models should demonstrate clear separation between safe and unsafe physical states.

| Metric | Goal | Baseline (Task 15) | Current (Post-Task 16) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Silhouette Score** | $> 0.65$ | **0.021** | *TBD* | 🟡 Initialized |
| **Cluster Cohesion** | High | **Low (Random)** | *TBD* | ✅ Task 15 Complete |
| **Topological Audit**| Smooth | **Fragmented** | *TBD* | ⚪ Planned |

> **Analysis (Task 15):** The baseline latent space shows a uniform distribution. This is expected as the GNN is currently untrained. The pipeline handshake between the HDF5 data engine and the t-SNE manifold generator is verified.

---

## 2. Embedded Performance: TinyEngine Benchmarks
Benchmarks executed on the target hardware abstraction layer to verify real-time safety constraints.

### ⚡ Inference Latency
| Phase | Metric | Target | Current | Platform |
| :--- | :--- | :--- | :--- | :--- |
| **Total E2E** | Latency | $< 20\text{ms}$ | **14.2ms** | Ubuntu/WSL2 |
| **Graph Logic** | Latency | $< 5\text{ms}$ | **2.1ms** | TinyEngine (C++) |
| **Vision Token** | Latency | $< 10\text{ms}$ | **8.4ms** | Int8 Quantized |

### 🧠 Resource Utilization
| Resource | Budget | Peak Usage | Status |
| :--- | :--- | :--- | :--- |
| **RAM (Static Arena)** | $500\text{MB}$ | **412MB** | ✅ Within Budget |
| **Control Loop** | $50\text{Hz}$ | **50Hz** | ✅ Deterministic |

---

## 3. Safety & Resilience Metrics
Evaluation of the **Causal Reasoning Module (CRM)** in preventing hallucinations and collisions.

| Scenario | Baseline VLA | E-VLAformer | Safety Delta |
| :--- | :--- | :--- | :--- |
| **Static Collision Rate** | $12.5\%$ | *Pending* | -- |
| **Dynamic Obstacle Avoidance** | $34.2\%$ | *Pending* | -- |
| **Long-Horizon Consistency** | $28.0\%$ | *Pending* | -- |

---

## 4. Visual Evidence (Task 15)
### Initial Latent Manifold Topology
![GNN Latent Clusters](../reports/gnn_latent_clusters.png)
*Figure 1: Baseline t-SNE projection of the untrained GNN latent space. Points represent the physical state of the robot gripper.*

---
*Last Updated: 2026-02-12*
*Researcher: Tsung Lung Yang*