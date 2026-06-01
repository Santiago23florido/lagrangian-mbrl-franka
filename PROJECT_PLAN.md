# PROJECT PLAN — Physics-Guided Model-Based RL for the Franka Panda (Isaac Lab)

**Working title:** *Sample-Efficient Model-Based RL with Lagrangian Dynamics Priors for Robotic Manipulation*

**Author:** Santiago
**Last updated:** 2026-06-01
**Target venues:** L4DC (primary, workshop/main track), CoRL (main track)
**Effort budget:** ~30 h/week over ~6 months (≈ 780 h total)

---

## 1. Problem Statement and the Gap

### 1.1 The setting
We want a robot arm (the 7-DoF Franka Emika Panda) to learn manipulation tasks
(starting with **reach**, then extending to **push/peg-insertion**) using
**model-based reinforcement learning (MBRL)**. In MBRL, the agent learns a
predictive model of the environment's dynamics and uses it for planning and/or
policy optimization, which is typically far more sample-efficient than
model-free RL because each real (or simulated) transition is reused many times
inside the learned model.

### 1.2 The gap
The accuracy and generalization of the learned dynamics model is the bottleneck
for MBRL performance. The dominant practice is to learn dynamics with an
**unstructured** function approximator (an ensemble of MLPs, e.g. PETS/MBPO).
These models:

- ignore known structure of rigid-body dynamics (the system is governed by the
  Euler–Lagrange equations with a symmetric positive-definite mass matrix),
- can violate energy conservation and produce physically implausible rollouts,
- and therefore extrapolate poorly and require more data to reach a given
  accuracy.

**Deep Lagrangian Networks (DeLaN)** and related physics-structured networks
(Hamiltonian Neural Networks, Lagrangian Neural Networks) embed the
Euler–Lagrange structure directly into the model, guaranteeing a physically
consistent inductive bias. They have been studied mostly for **forward/inverse
dynamics learning in isolation**, on low-DoF systems, and largely **outside a
closed RL loop**.

> **The precise gap:** There is no rigorous, empirically-validated study of
> whether a Lagrangian-structured dynamics model, used *as the predictive model
> inside an MBRL loop*, improves **sample efficiency** for a realistic
> high-DoF manipulator (Franka, 7-DoF) — and no accompanying **sample-complexity
> analysis** that explains *why* the physical prior helps, quantified in terms
> of the reduced hypothesis-class complexity.

### 1.3 Claim we want to defend
> Embedding Lagrangian structure into the dynamics model reduces the effective
> complexity of the model class, which (i) provably tightens the model-error /
> sample-complexity bound for MBRL, and (ii) empirically yields better
> sample efficiency and more stable long-horizon rollouts than model-free and
> unstructured model-based baselines on Franka manipulation in Isaac Lab.

---

## 2. Theoretical Contribution to Aim For

The goal is a **sample-complexity bound for MBRL that is explicitly improved by
the structural prior**. Concretely:

### 2.1 Form of the result
A bound of the shape

```
J(π*) − J(π̂)  ≤  C · H² · ε_model  + (statistical / optimization terms)
```

where `ε_model` is the one-step (or multi-step) model error of the learned
dynamics, `H` is the planning horizon, and `π̂` is the policy optimized in the
learned model. This is the standard "simulation lemma" / model-based policy
optimization style decomposition (cf. MBPO, Luo et al. 2019). Our contribution
is **bounding `ε_model` via the statistical complexity of the model class** and
showing that the Lagrangian-structured class is provably smaller.

### 2.2 The key step — complexity of the model class
Use a generalization bound (Rademacher complexity / covering numbers / metric
entropy) for regression with the structured hypothesis class `H_DeLaN` versus
the unstructured class `H_MLP`:

```
ε_model(H)  ≲  inf_{f∈H} ε_approx(f)  +  Õ( Comp(H) / √N )
```

- **Approximation term:** For rigid-body systems the true dynamics *lie in*
  (or arbitrarily close to) the Lagrangian class, so `ε_approx(H_DeLaN)` is small
  by construction, whereas the MLP pays no structural penalty but must learn the
  constraint from data.
- **Estimation term:** The structured class encodes the constraint that the
  mass matrix `M(q)` is symmetric positive-definite and that forces derive from
  a scalar Lagrangian, restricting the function class and lowering its
  complexity `Comp(H_DeLaN) ≤ Comp(H_MLP)`, hence lowering the `1/√N`
  estimation error.

The headline statement: **for a target model error `ε`, the structured class
needs `N_DeLaN = O(N_MLP / κ)` samples**, where `κ ≥ 1` quantifies the capacity
reduction from the physical constraints.

### 2.3 Assumptions to make explicit (and to relax later)
- **A1.** True dynamics are (approximately) those of a rigid-body Lagrangian
  system: `M(q) q̈ + c(q, q̇) + g(q) = τ`, with `M ≻ 0` and smooth.
- **A2.** Bounded states/torques on a compact set; Lipschitz dynamics.
- **A3.** I.i.d. (or β-mixing) transition samples for the regression bound; note
  the on-policy data in RL violates i.i.d. — handle via a mixing assumption or a
  data-aggregation / iterative argument.
- **A4.** Realizable or agnostic setting stated clearly.
- **A5.** (For the policy bound) bounded rewards, known/learned reward, finite
  horizon `H`.

**Honest scope:** The cleanest defensible result is likely a bound *conditional*
on A1–A5 with `κ` made concrete for a specific parameterization of `M(q)` (e.g.
Cholesky-factor network as in DeLaN). A fully tight, distribution-free bound is a
stretch goal; a clear conditional bound + matching empirical sample-efficiency
curves is a strong, publishable contribution on its own.

---

## 3. Reading List (foundational papers)

> Maintain full BibTeX in `docs/references.bib` as you read. Annotate each entry
> with a one-paragraph "what we take from it" note.

### 3.1 Physics-structured dynamics learning
- **Lutter, Ritter, Peters (2019)** — *Deep Lagrangian Networks: Using Physics
  as Model Prior for Deep Learning* (ICLR). **Core method paper.**
- **Lutter & Peters (2023)** — *Combining Physics and Deep Learning for
  Continuous-Time Dynamics Models* (journal extension of DeLaN).
- **Greydanus, Dzamba, Yosinski (2019)** — *Hamiltonian Neural Networks* (NeurIPS).
- **Cranmer et al. (2020)** — *Lagrangian Neural Networks* (ICLR DeepDiffEq ws).
- **Zhong, Dey, Chakraborty (2020)** — *Symplectic ODE-Net* (ICLR); learning
  Hamiltonian dynamics with control.
- **Saemundsson et al. (2020)** — *Variational Integrator Networks*.
- **Gupta et al. (2020)** — *Structured Mechanical Models* (L4DC).

### 3.2 Model-based RL with learned dynamics
- **Deisenroth & Rasmussen (2011)** — *PILCO* (ICML). Data-efficient GP-based MBRL.
- **Chua et al. (2018)** — *PETS: Deep RL in a Handful of Trials...* (NeurIPS).
- **Janner et al. (2019)** — *MBPO: When to Trust Your Model* (NeurIPS).
- **Hafner et al. (2019/2020/2023)** — *PlaNet / Dreamer / DreamerV3*.
- **Nagabandi et al. (2018)** — *Neural Network Dynamics for Model-Based Deep RL
  with Model-Free Fine-Tuning* (ICRA).
- **Williams et al. (2017)** — *MPPI* (model-predictive path integral control).

### 3.3 Sample-complexity / theory for MBRL
- **Kearns & Singh (2002)** — *Near-Optimal RL in Polynomial Time* (simulation lemma roots).
- **Luo et al. (2019)** — *Algorithmic Framework for MBRL with Theoretical
  Guarantees* (SLBO).
- **Sun et al. (2019)** — *Model-based RL in contextual decision processes:
  PAC bounds and exponential improvements over model-free*.
- **Agarwal, Jiang, Kakade, Sun** — *Reinforcement Learning: Theory and
  Algorithms* (monograph; simulation lemma, PAC-MDP, value-difference).
- **Tu & Recht (2019)** — *Sample complexity of LQR: model-based vs model-free*
  (clean separation result; template for our argument).
- **Wainwright (2019)** — *High-Dimensional Statistics* (Rademacher/metric
  entropy tools).
- **Bartlett & Mendelson (2002)** — *Rademacher and Gaussian complexities*.

### 3.4 Simulation / Isaac Lab
- **Mittal et al. (2023)** — *Orbit / Isaac Lab: A Unified Simulation Framework
  for Interactive Robot Learning Environments* (RA-L).
- **Makoviychuk et al. (2021)** — *Isaac Gym: High-Performance GPU-Based Physics
  Simulation*.
- **Rudin et al. (2022)** — *Learning to Walk in Minutes Using Massively
  Parallel Deep RL* (RSL-RL; PPO at scale).
- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
- SKRL docs: https://skrl.readthedocs.io/

---

## 4. Phased Timeline (~6 months, ~30 h/week)

> Milestones are written so each is independently demonstrable. Dates are
> anchored to the start date **2026-06-01**.

### Phase 0 — Setup & reproduction (Weeks 1–2, by **2026-06-14**) ⭐ FIRST DELIVERABLE
- Install Isaac Sim + Isaac Lab (2.x) on the RTX 4070; verify a headless
  Franka environment runs.
- **Reproduce a Franka-reach baseline** with RSL-RL or SKRL PPO; log a learning
  curve, confirm it solves reach.
- **Scaffold the Lagrangian network** (`deep_lagrangian_network.py`) — forward
  pass producing `M(q)`, gradients of the Lagrangian, and predicted `q̈`; train
  it offline on logged Franka transitions to predict accelerations (sanity:
  loss decreases, `M(q) ≻ 0`).
- **Exit criteria:** PPO reach curve reproduced + DeLaN trains on offline data
  and beats an MLP on held-out one-step acceleration MSE.

### Phase 1 — Theory minimal viable result (Weeks 3–6, by **2026-07-12**)
- Write the simulation-lemma decomposition for our setting (`theory/`).
- Derive the complexity gap `Comp(H_DeLaN) ≤ Comp(H_MLP)` for the chosen `M(q)`
  parameterization; pin down `κ` and all assumptions.
- Produce a clean conditional sample-complexity statement + proof sketch.
- **Exit criteria:** A self-contained ~4-page theory note that a co-author/advisor
  signs off on; identify which assumptions the experiments must probe.

### Phase 2 — Minimal MBRL implementation (Weeks 7–10, by **2026-08-09**)
- Implement the model-based loop (`mbrl/training_loop.py`): collect → train
  dynamics → plan/optimize policy in model → act. Start with MPC (MPPI/CEM) on
  the learned model for reach (no policy network needed first).
- Plug in both dynamics models (DeLaN and MLP-ensemble) behind a common
  interface so they're swappable.
- **Exit criteria:** MBRL-with-DeLaN solves reach; produce the first
  sample-efficiency comparison (return vs. environment steps) DeLaN vs MLP vs PPO.

### Phase 3 — Scaling & robustness (Weeks 11–16, by **2026-09-20**)
- Move to policy optimization in-model (MBPO-style Dyna) if MPC is too slow.
- Add contact-rich task (push or peg-insertion) to stress A1 (pure-Lagrangian
  assumption breaks under contact) — important honesty test.
- Engineering: vectorized envs, checkpointing, config system, seeds, W&B.
- **Exit criteria:** Method runs on ≥2 tasks, ≥5 seeds each, within 12GB VRAM.

### Phase 4 — Full experiments & ablations (Weeks 17–22, by **2026-11-01**)
- Run the full benchmark matrix (Section 5), ≥5 seeds, with confidence intervals.
- Ablations (Section 5.3). Generate all paper figures from logged data
  reproducibly (`scripts/` → `figures/`).
- **Exit criteria:** Frozen results; figures regenerate from raw logs via script.

### Phase 5 — Writing & submission (Weeks 23–26, by **2026-11-30**)
- Draft paper; align empirical curves with the theory's predicted `κ`.
- Internal review, polish, submit to workshop first (Section 8).
- **Exit criteria:** Submitted manuscript + released code + reproducibility README.

---

## 5. Baselines, Metrics, and Ablations

### 5.1 Baselines
| Class | Method | Why |
|---|---|---|
| Model-free | **PPO** (RSL-RL/SKRL) | Standard Isaac Lab baseline; sample-inefficient reference. |
| Model-free | **SAC** | Off-policy, more sample-efficient MF reference. |
| Model-based (unstructured) | **PETS / MLP-ensemble + MPC** | The key "does structure help?" comparison. |
| Model-based (unstructured) | **MBPO** | Strong unstructured MBRL. |
| Model-based (structured) | **Ours: DeLaN + same MBRL loop** | Isolates the effect of the prior. |
| (Optional) | **HNN/LNN variant** | Show DeLaN choice isn't arbitrary. |

The decisive comparison is **DeLaN vs MLP-ensemble inside the *same* MBRL loop** —
same planner, same hyperparameters, only the dynamics class changes.

### 5.2 Metrics
- **Sample efficiency:** task return vs. environment steps; **steps-to-threshold**
  (env steps to reach X% success). Primary headline metric.
- **Asymptotic performance / success rate.**
- **Model accuracy:** one-step and H-step rollout MSE on held-out transitions.
- **Physical consistency:** energy drift over rollout; `M(q)` positive-definiteness.
- **Compute:** wall-clock and VRAM (single 4070 constraint).
- Report with ≥5 seeds, mean ± 95% CI; use stratified bootstrap (rliable).

### 5.3 Ablations
- Structure on/off (DeLaN vs MLP) — the core ablation.
- `M(q)` parameterization (Cholesky vs diagonal vs full).
- With/without learned dissipation & input/actuation model.
- Ensemble size; planning horizon `H`; model-update frequency.
- Data regime: vary `N` to **trace the empirical sample-complexity curve** and
  compare its slope to the theory's predicted `κ`.
- Contact task: quantify degradation when A1 is violated.

---

## 6. Experiment Tracking & Reproducibility

- **Config:** Hydra/YAML (`configs/`); every run dumps its resolved config.
- **Tracking:** Weights & Biases (primary) + TensorBoard; log seeds, git SHA,
  config hash, hardware.
- **Seeds:** fix and record; sweep ≥5 seeds per cell.
- **Determinism:** set torch/cuda/numpy seeds; note Isaac Sim non-determinism and
  control for it by seed-averaging.
- **Artifacts:** checkpoints + eval logs versioned; raw logs → figures via a
  single script so every figure is regenerable.
- **Environment capture:** pinned `requirements.txt`/`pyproject.toml`, recorded
  Isaac Lab/Sim versions, `pip freeze` snapshot saved per experiment batch.
- **Code review gates:** unit tests for the DeLaN forward pass (PD mass matrix,
  energy conservation on a known pendulum), metric computations.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Isaac Lab install / GPU (12GB) friction | High | Med | Time-box to Phase 0; headless + small env counts; fall back to a MuJoCo Franka for early dev if blocked. |
| Contact dynamics violate the Lagrangian prior (A1) | High | High | Frame contact as the *honesty experiment*; add learned residual/dissipation term; scope main claim to smooth-dynamics regime. |
| Theory bound not tight / not novel | Med | High | Aim for a clean *conditional* separation (LQR-style, Tu & Recht template); de-risk by getting Phase-1 advisor sign-off early. |
| DeLaN training instability (Cholesky, 2nd derivatives) | Med | Med | Use stable parameterization, gradient clipping, double precision for the mass matrix; unit-test on pendulum/2-link arm first. |
| MPC too slow on 4070 | Med | Med | Switch to Dyna/MBPO-style in-model policy learning; reduce horizon; amortize planner. |
| Scope creep (too many tasks) | Med | Med | Reach is the must-have; push/insertion are stretch; freeze scope at Phase 3. |
| Negative result (structure doesn't help on Franka) | Low–Med | High | Still publishable: a careful negative result + theory on *when* structure helps is valuable; pre-register the comparison. |
| Single-developer time risk | Med | Med | Each phase is independently demonstrable; protect Phase 0/1 as the minimum publishable unit. |

---

## 8. Suggested Submission Path

1. **Workshop first (feedback):** target an L4DC or NeurIPS/ICML workshop
   (e.g. on physics-informed ML / robot learning). Submit the Phase-0–2 result
   (theory sketch + reach sample-efficiency curves). Goal: reviewer feedback,
   sanity-check the framing, and de-risk before a main-track deadline.
2. **Main venue:** **L4DC** is the natural fit (learning + dynamics + theory).
   Submit the full result: conditional sample-complexity bound + multi-task,
   multi-seed sample-efficiency study + ablations tracing the empirical `κ`.
3. **Backup / extension:** **CoRL** if the robotics/manipulation story (contact
   tasks, real-arm transfer as future work) is the stronger angle, or if timing
   misses L4DC.

**Minimum publishable unit:** reproduced reach baseline + DeLaN-in-MBRL beating
MLP-in-MBRL on sample efficiency (≥5 seeds) + the conditional theory bound.
Everything beyond that (contact tasks, extra ablations, real-robot future work)
strengthens but is not required for a first submission.

---

## 9. Immediate Next Actions (this week)
- [ ] Install Isaac Sim + Isaac Lab 2.x; confirm GPU works headless.
- [ ] Run stock Franka-reach PPO; save the learning curve to `logs/`.
- [ ] Log a dataset of Franka transitions `(q, q̇, τ, q̈)` for offline dynamics fitting.
- [ ] Fill in `deep_lagrangian_network.py` forward pass; unit-test PD mass matrix.
- [ ] Start `docs/references.bib`; read DeLaN + MBPO + Tu & Recht closely.
