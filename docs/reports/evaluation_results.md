# 📊 E-VLAformer: Evaluation & Benchmarks

This report tracks the quantitative performance of the **E-VLAformer** architecture. It serves as the primary evidence for the NeurIPS 2026 submission, documenting the transition from raw simulation to a safety-aware Neuro-Symbolic world model.

---

## 1. World Model Quality: Latent Space Topology
We evaluate the Graph World Model (GWM) by analyzing its latent space. High-quality world models must demonstrate clear separation between semantic classes or perfect identity preservation.

| Metric | Goal | Baseline (Task 15) | Current (Post-Task 20) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Manifold Scale** | $> 2.0x$ Expansion | **150 Units** | **400 Units** | ✅ **2.6x Verified** |
| **Silhouette Stability**| $\approx 0.00$ (Identity) | **0.28** | **0.0000** | ✅ **Identity Mapping** |
| **Embedding Variance** | $\sigma^2 > 0.1$ | **N/A** | **0.5309** | ✅ **Rich Features** |

---

## 2. Action Policy & Sensor Feedback (Phase 3)
Evaluation of the VLA Policy Head and the high-fidelity proprioception loop.

### 🦾 Task 21/22: Action & Proprioception Loop
| Metric | Dimension / Logic | Activation / Filter | Status |
| :--- | :--- | :--- | :--- |
| **Input Fusion** | 548-dim | LayerNorm | ✅ Verified |
| **Proprioception Map**| 4-DOF Normalized | **Alpha-Filter ($\alpha=0.7$)** | ✅ **Certified** |
| **Numerical Safety** | Unit Range | [-1.0, 1.0] | ✅ **Clamping Verified** |
| **Inference Test** | $\Delta$ Joint Vector | Tanh | ✅ **Smoke Test Passed** |

> **Verification Note (Task 22):** The Proprioception Handler successfully transformed noisy raw input `[45.2, -44.8, 0.5, 10.1]` into the stable latent tensor `[[0.5022, -0.4977, 0.0055, 0.1122]]`. The integration of the low-pass filter ensures that the Policy Head receives a stabilized signal, critical for preventing MG996R servo oscillation.

---

## 3. Cognitive Resilience: Object Permanence & Occlusion

### 🧠 Persistence Benchmarks (Task 17 - 20)
| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Persistence Duration** | $> 500\text{ frames}$ | **1,000+ Frames** | ✅ Logic Verified |
| **Latent Drift** | $< 5.0 \%$ | **0.0%** | ✅ **Task 19 Verified** |
| **Edge Case Resilience** | Hardened | **10% Blink Rate** | ✅ Task 18 Complete |
| **Phase 2 Freeze** | Certified | **certified_gwm_v1** | ✅ **Task 20 Locked** |

---

## 4. Embedded Performance: TinyEngine Benchmarks
Benchmarks executed on the target hardware abstraction layer to verify real-time safety constraints.

### ⚡ Inference Latency
| Phase | Metric | Target | Current | Platform |
| :--- | :--- | :--- | :--- | :--- |
| **Total E2E** | Latency | $< 20\text{ ms}$ | **14.2ms** | Ubuntu/WSL2 |
| **Graph Logic** | Latency | $< 5\text{ ms}$ | **2.1ms** | TinyEngine (C++) |
| **Vision Token** | Latency | $< 10\text{ ms}$ | **8.4ms** | Int8 Quantized |

---

## 5. Visual Evidence (Evolution)

### 5.1 Latent Manifold Comparison (Task 15 vs Task 20)

| Untrained Baseline (Task 15) | Post-Contrastive Identity (Task 20) |
| :---: | :---: |
| ![Baseline](../reports/task15_baseline.png) | ![Trained](../reports/task16_trained.png) |
| *Scale: 150 | Random Nebula* | *Scale: 400 | Stable Identity Cluster* |



---
*Last Updated: 2026-02-13* *Researcher: Tsung Lung Yang*