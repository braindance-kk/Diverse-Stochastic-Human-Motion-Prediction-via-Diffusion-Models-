# Stochastic Human Motion Prediction via Diffusion Models with Weakly-Supervised Action Transitions

> **Honours Project · Australian National University · School of Computing · 2024–2025**

This project studies **stochastic human motion prediction**: given a short observed motion history and a sequence of future action labels, the model generates **multiple plausible future human-motion trajectories** rather than a single deterministic continuation.

The project focuses on two major challenges in human motion generation:

- **Multimodal futures:** the same motion history can lead to multiple valid future actions and trajectories.
- **Cross-action transitions:** real human behavior contains frequent transitions between actions, while densely annotated transition data is expensive and incomplete.

To address these problems, the project combines a **Transformer-based conditional diffusion model** with **weakly-supervised action-transition learning**, enabling diverse, action-conditioned, and temporally coherent motion prediction.

---

## Overview

Traditional deterministic motion-prediction methods usually produce one “most likely” future and therefore struggle to represent the inherent uncertainty of human behavior. Action-conditioned motion generators can produce realistic isolated actions, but often fail when several actions must be connected smoothly.

This project introduces an action-driven stochastic prediction framework with the following pipeline:

```text
Observed Motion History
        │
        ├──────────────┐
        │              │
        ▼              ▼
 History Encoder   Action Label
        │              │
        └──────┬───────┘
               ▼
       Condition Embedding
               │
               ▼
Gaussian Noise → Transformer Diffusion Model
               │
               ▼
       Iterative Denoising
               │
               ▼
 Multiple Plausible Future Motions
```

Only the future motion is corrupted during the forward diffusion process. The observed history remains clean and is used together with the target action label as conditioning information throughout reverse denoising.

---

## Key Features

### 1. Conditional Diffusion for Stochastic Motion Prediction

The model uses an **encoder-only Transformer** as the denoising network. It is conditioned on:

- observed historical motion;
- diffusion timestep;
- future action label.

During inference, generation starts from Gaussian noise and progressively reconstructs a future motion sequence. Sampling with different noise realizations allows the same history/action condition to produce multiple plausible futures.

### 2. SMPL + Rot6D Motion Representation

Human pose is represented using **SMPL pose parameters**. Joint rotations are converted to the continuous **6D rotation representation (Rot6D)** during training.

Rot6D provides a smoother optimization space than Euler angles or quaternions and avoids issues such as gimbal lock and rotation discontinuities. Predictions are converted back to SMPL-compatible rotations for evaluation and visualization.

### 3. Weakly-Supervised Action-Transition Learning

Real datasets cannot contain every possible action pair or accurately label every transition frame. Instead of requiring dense transition annotations, the training pipeline constructs cross-action examples by combining motion fragments from different action categories.

Candidate future sequences are selected according to pose similarity near the transition boundary. A transition offset is estimated from the distance between the final historical pose and the first pose of the candidate future action.

A **DCT-based temporal smoothness prior** suppresses high-frequency motion changes around the transition region and encourages more natural connections between actions.

The overall training objective combines:

```text
L = λ_rec · L_rec + λ_smooth · L_smooth + L_simple
```

where:

- `L_rec` supervises reconstruction of the future motion;
- `L_smooth` encourages smooth cross-action transitions;
- `L_simple` is the diffusion denoising objective.

### 4. Classifier-Free Guidance

During training, action conditions are randomly masked so that the network learns both conditional and unconditional denoising.

At inference time, **classifier-free guidance (CFG)** controls the trade-off between:

- stronger adherence to the requested action semantics;
- greater stochastic diversity.

### 5. Variable-Length Motion Prediction

The project also explores motion sequences whose duration is not fixed in advance.

During training, repeated terminal frames are appended to motion sequences to teach the network an implicit stopping behavior. During inference, generation can stop when motion variation over a recent temporal window falls below a threshold, with a maximum generation horizon used as a safeguard.

---

## Model Architecture

The diffusion denoising network follows a compact Transformer architecture:

```text
History Motion ── Linear Projection ───────────────┐
                                                   │
Diffusion Step ── MLP ──┐                         │
                        ├─ Condition Token ────────┤
Action Label ──── MLP ──┘                         │
                                                   ▼
Noised Future ── Linear + Positional Encoding → Transformer Encoder
                                                   │
                                                   ▼
                                             Linear Layer
                                                   │
                                                   ▼
                                         Predicted Clean Motion
```

The architecture gives the model access to both long-range temporal context and high-level action semantics while denoising the future sequence.

---

## Datasets

The experiments use motion data following the preprocessing protocol used by the weakly-supervised transition baseline.

| Dataset | Motion Length | Train | Test | Actions |
|---|---:|---:|---:|---:|
| GRAB | 100–501 | 1,149 | 319 | 4 |
| HumanAct12 | 35–290 | 727 | 197 | 12 |
| NTU RGB-D subset | 35–201 | 3,399 | 361 | 13 |

The thesis also discusses BABEL as a dataset containing explicit action-transition samples.

Typical preprocessing includes frame-rate normalization, removal of global root translation, filtering short clips, and conversion to a common pose representation.

---

## Evaluation

The framework evaluates generated motion from several complementary perspectives:

- **FID** — distribution-level realism;
- **Action Classification Accuracy** — semantic consistency with the target action;
- **Diversity / DTW-aligned Diversity** — variation among generated futures;
- **ADE / DTW-aligned ADE** — prediction accuracy relative to reference motion.

The model is compared with representative stochastic human-motion generation approaches including **Action2Motion**, **ACTOR**, **DLow**, and the weakly-supervised action-transition baseline (**WATL**).

---

## Experimental Highlights

On the GRAB evaluation, the diffusion model demonstrates particularly strong stochastic diversity. The reported diversity scores are higher than the compared baselines:

| Method | Divw ↑ | Div ↑ |
|---|---:|---:|
| Action2Motion | 0.50 | 0.76 |
| DLow | 0.74 | 0.92 |
| ACTOR | 1.06 | 1.04 |
| WATL | 1.10 | 1.37 |
| **Ours** | **1.47** | **2.10** |

The experiments therefore demonstrate the model's ability to generate substantially different futures from the same history and action condition.

However, trajectory accuracy remains a limitation. On GRAB, the model reports `ADE = 2.73 ± 0.03` and `ADEw = 2.70 ± 0.02`, indicating that improving local motion structure and transition smoothness remains important.

---

## Limitations

The current implementation has several limitations:

- the encoder-only Transformer has limited structural bias for fine-grained skeletal motion;
- weak supervision introduces uncertainty around exact transition timing;
- complex human-object interactions are modeled using human pose alone, without explicit object state or scene context;
- Transformer-based modeling is sensitive to noisy pose estimates in datasets such as HumanAct12 and NTU RGB-D;
- transitions between kinematically dissimilar actions can still contain discontinuities.

---

## Future Work

Potential extensions include:

- skeleton-aware or graph-based architectures;
- stronger transition-specific constraints;
- velocity and acceleration continuity losses;
- explicit hand-object/contact representations;
- physics-informed motion constraints;
- improved robustness to noisy pose estimation;
- richer conditioning signals such as text, environment context, and interaction information.

---

## Applications

This work is relevant to applications including:

- human behavior forecasting;
- autonomous systems and robotics;
- human–robot interaction;
- VR/AR avatars;
- game and film character animation;
- controllable motion synthesis.

---

## Project Information

**Project:** Stochastic Human Motion Prediction via Diffusion Models with Weakly-Supervised Action Transitions  
**Institution:** Australian National University — School of Computing  
**Type:** Honours Project  
**Period:** 2024–2025  

---

## Acknowledgements

This project builds upon research in diffusion-based motion generation, stochastic human motion prediction, SMPL-based pose modeling, and weakly-supervised action-transition learning.

