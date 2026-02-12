# 📊 E-VLAformer: Evaluation & Benchmarks

This report tracks the quantitative performance of the **E-VLAformer** architecture. It serves as the primary evidence for the NeurIPS 2026 submission, documenting the transition from raw simulation to a safety-aware Neuro-Symbolic world model.

---

## 1. World Model Quality: Latent Space Topology
We evaluate the Graph World Model (GWM) by analyzing its latent space. High-quality world models must demonstrate clear separation between safe and unsafe physical states.

| Metric | Goal | Baseline (Task 15) | Current (Post-Task 16) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Manifold Scale** | $> 2.0x$ Expansion | **150 Units** | **400 Units** | ✅ **2.6x Verified** |
| **Silhouette Score** | $> 0.55$ | **0.021** | **0.42** | 🟡 Improving |
| **Topological Audit**| Smooth | **Fragmented** | **Clustered** | ✅ Task 16 Complete |

---

## 2. Cognitive Resilience: Object Permanence (Task 17 Verified)
We evaluate the system's ability to maintain a stable world state during high-entropy occlusion events (e.g., objects hidden behind lids).



### 🧠 Persistence Benchmarks
| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Persistence Duration** | $> 500\text{ frames}$ | **1,000+ Frames** | ✅ Logic Verified |
| **Latent Feature Drift** | $0.0\%$ | **0.0%** | ✅ Deterministic |
| **Node Recovery Rate** | $> 95\%$ | **100% (Stress Test)** | ✅ Verified |

> **Analysis (Task 17):** The implementation of the **Graph Memory Buffer** has successfully eliminated "Causal Hallucinations" during occlusion. In stress tests, the GNN maintained 100% feature parity over 1,000 frames of darkness, ensuring the robot retains a "Mental Map" of objects it can no longer see.

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
| **Occlusion Resilience** | $15.0\%$ | **98.5% (Logic)** | ✅ **+83.5% Improvement** |

---

## 5. Visual Evidence (Evolution)

### 5.1 Latent Manifold Comparison (Task 15 vs Task 16)


| Untrained Baseline (Task 15) | Post-Contrastive Training (Task 16) |
| :---: | :---: |
| ![Baseline](../reports/task15_baseline.png) | ![Trained](../reports/task16_trained.png) |
| *Scale: 150 | Random Nebula* | *Scale: 400 | Structured Features* |

---
*Last Updated: 2026-02-12* *Researcher: Tsung Lung Yang*