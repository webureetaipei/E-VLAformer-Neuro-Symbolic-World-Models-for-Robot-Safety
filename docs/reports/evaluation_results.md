# 📊 E-VLAformer: Evaluation & Benchmarks

This report tracks the quantitative performance of the **E-VLAformer** architecture. It serves as the primary evidence for the NeurIPS 2026 submission, documenting the transition from raw simulation to a safety-aware Neuro-Symbolic world model.

---

## 1. World Model Quality: Latent Space Topology
We evaluate the Graph World Model (GWM) by analyzing its latent space. High-quality world models must demonstrate clear separation between safe and unsafe physical states.

| Metric | Goal | Baseline (Task 15) | Current (Post-Task 16) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Manifold Scale** | $> 2.0x$ Expansion | **150 Units** | **400 Units** | ✅ **2.6x Verified** |
| **Silhouette Score** | $> 0.55$ | **0.021** | **0.42** | 🚀 Task 19 Active |
| **Topological Audit**| Smooth | **Fragmented** | **Clustered** | ✅ Task 16 Complete |

---

## 2. Cognitive Resilience: Object Permanence & Occlusion

### 🧠 Persistence Benchmarks (Task 17 & 18)
| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Persistence Duration** | $> 500\text{ frames}$ | **1,000+ Frames** | ✅ Logic Verified |
| **Edge Case Resilience** | Hardened | **10% Blink Rate** | ✅ Task 18 Complete |
| **Node Recovery Rate** | $> 95\%$ | **100% (Stress Test)** | ✅ Verified |

> **Analysis (Task 18):** We have moved beyond pure logic tests. By implementing **Blink Logic** in Isaac Sim 4.5.0, we successfully generated a **Hardened Dataset** (`task18_occlusion_test_001.h5`). The system now demonstrates 100% recovery of "Vanishing Nodes" by cross-referencing the Graph Memory Buffer when $P(\text{visibility}) \rightarrow 0$.

---

## 3. Embedded Performance: TinyEngine Benchmarks
Benchmarks executed on the target hardware abstraction layer to verify real-time safety constraints.

### ⚡ Inference Latency
| Phase | Metric | Target | Current | Platform |
| :--- | :--- | :--- | :--- | :--- |
| **Total E2E** | Latency | $< 20\text{ ms}$ | **14.2ms** | Ubuntu/WSL2 |
| **Graph Logic** | Latency | $< 5\text{ ms}$ | **2.1ms** | TinyEngine (C++) |
| **Vision Token** | Latency | $< 10\text{ ms}$ | **8.4ms** | Int8 Quantized |

### 🧠 Resource Utilization
| Resource | Budget | Peak Usage | Status |
| :--- | :--- | :--- | :--- |
| **RAM (Static Arena)** | $500\text{ MB}$ | **412MB** | ✅ Within Budget |
| **Control Loop** | $50\text{ Hz}$ | **50Hz** | ✅ Deterministic |

---

## 4. Safety & Resilience Metrics (Phase 3 Targets)
Evaluation of the **Causal Reasoning Module (CRM)** in preventing hallucinations and collisions.

| Scenario | Baseline VLA | E-VLAformer | Safety Delta |
| :--- | :--- | :--- | :--- |
| **Static Collision Rate** | $12.5\%$ | *Pending Task 19* | -- |
| **Dynamic Obstacle Avoidance** | $34.2\%$ | *Pending Task 19* | -- |
| **Occlusion Resilience** | $15.0\%$ | **98.5% (Hardened)**| ✅ **+83.5% Improvement** |

---

## 5. Visual Evidence (Evolution)

### 5.1 Latent Manifold Comparison (Task 15 vs Task 16)


| Untrained Baseline (Task 15) | Post-Contrastive Training (Task 16) |
| :---: | :---: |
| ![Baseline](../reports/task15_baseline.png) | ![Trained](../reports/task16_trained.png) |
| *Scale: 150 | Random Nebula* | *Scale: 400 | Structured Features* |

---
*Last Updated: 2026-02-12* *Researcher: Tsung Lung Yang*