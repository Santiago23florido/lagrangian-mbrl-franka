# Theory & Derivations

Working notes toward the paper's theoretical contribution: a sample-complexity
bound for MBRL that is *provably tightened* by the Lagrangian structural prior.
This is a living document; the polished version becomes the paper's theory
section. See `PROJECT_PLAN.md` §2 for the high-level plan.

---

## 1. Setting and notation

Finite-horizon MDP with horizon `H`, states `s = (q, q̇) ∈ S ⊂ R^{2d}`, actions
(torques) `a = τ ∈ A ⊂ R^d`, deterministic-ish transition given by the
rigid-body dynamics integrated over `dt`, reward `r(s, a)` bounded in `[0, 1]`,
policy value `J(π) = E[ Σ_{t<H} r(s_t, a_t) ]`.

True one-step dynamics `f*`; learned model `f̂ ∈ H` fit on `N` transitions.
Model error (define precisely; pick one and stay consistent):

    ε_model(f̂) := E_{(s,a)∼ρ} ‖ f̂(s,a) − f*(s,a) ‖²        (one-step, under data dist ρ)

---

## 2. Step A — policy suboptimality in terms of model error (simulation lemma)

Standard model-based decomposition (Kearns–Singh; Luo et al. 2019; Janner et al.
2019). For the policy `π̂` optimal in the learned model `f̂`,

    J(π*) − J(π̂)  ≤  C · H² · √(ε_model)   +   ε_opt   +   ε_dist                  (1)

where:
- `C` depends on the Lipschitz constant of the value function / reward (use A2),
- `H²` is the usual horizon amplification from compounding model error,
- `ε_opt` is the planner/policy-optimization suboptimality (assume small / set by
  the planner),
- `ε_dist` accounts for distribution shift between the data distribution `ρ` and
  the state distribution induced by `π̂` (handle via on-policy data aggregation
  or a concentrability/coverage assumption).

> TODO: nail the exact constant and whether it's `√ε_model` or `ε_model`
> depending on the chosen error metric; cite Agarwal–Jiang–Kakade–Sun for the
> value-difference lemma form we use.

The point of Step A: **performance is governed by `ε_model`.** Everything
interesting happens in how fast `ε_model` shrinks with `N` for each model class.

---

## 3. Step B — generalization bound on model error per class

For a hypothesis class `H` fit by ERM on `N` i.i.d. (or β-mixing) samples,
standard learning theory (Bartlett–Mendelson; Wainwright) gives

    ε_model(f̂_H)  ≲  ε_approx(H)  +  Õ( R_N(H) )                                   (2)

where `ε_approx(H) = inf_{f∈H} ε_model(f)` is the approximation error and
`R_N(H)` is the Rademacher complexity (or a covering-number / metric-entropy
proxy) of `H`, typically `R_N(H) ∝ Comp(H) / √N`.

> TODO: state the i.i.d./mixing assumption (A3) explicitly and, for the RL
> setting, the iterative/DAgger-style argument that lets us reuse a supervised
> bound despite on-policy data.

---

## 4. Step C — the structural gap (the crux)

Compare `H_DeLaN` (Lagrangian-structured) and `H_MLP` (unstructured) of matched
nominal size.

**Approximation.** Under A1 (true system is rigid-body Lagrangian), the true
dynamics are *realizable* (or nearly so) by the DeLaN class:

    ε_approx(H_DeLaN) ≈ 0.

The MLP can also approximate `f*` (universal approximation), so this term is not
where DeLaN wins asymptotically — but the MLP must *spend capacity* learning the
constraint structure that DeLaN gets for free.

**Estimation / complexity.** DeLaN constrains the model: the mass matrix is
`M(q) = L_θ(q) L_θ(q)ᵀ + εI` (symmetric PD by construction) and the conservative
forces derive from a single scalar potential `V_θ(q)`. This is a strict subset of
the input–output maps an MLP of comparable width can represent, so

    Comp(H_DeLaN)  ≤  Comp(H_MLP),    write   Comp(H_MLP) / Comp(H_DeLaN) =: κ ≥ 1.

> TODO: make `κ` concrete. Options:
>   (i) parameter-counting / effective-dimension argument for the Cholesky
>       parameterization (count free functions: d(d+1)/2 mass entries + 1
>       potential vs. 2d free outputs);
>   (ii) covering-number bound for the constrained class;
>   (iii) a clean linear/LQR surrogate (cf. Tu & Recht 2019) where the gap is
>       exactly computable, then argue it transfers qualitatively.

**Resulting sample complexity.** Combining (2) with the complexity gap, to reach
target model error `ε`,

    N_DeLaN  ≈  N_MLP / κ.                                                          (3)

Plugging into (1): the structured model reaches a target policy suboptimality
with a factor-`κ` fewer environment samples (up to the approximation and
distribution-shift terms).

---

## 5. Assumptions ledger (keep honest; map each to an experiment)

| ID | Assumption | Probed by |
|----|------------|-----------|
| A1 | True dynamics are rigid-body Lagrangian (`M≻0`, smooth) | contact-task degradation experiment |
| A2 | Bounded compact state/torque set; Lipschitz dynamics & value | sanity ranges in env wrapper |
| A3 | i.i.d. / β-mixing samples for the supervised bound | data-aggregation argument; seed-averaging |
| A4 | (Near-)realizability for DeLaN; agnostic for MLP | held-out model MSE floor |
| A5 | Bounded reward, finite horizon `H` | task definition |

---

## 6. What would make this tight vs. what is the safe target

- **Safe, publishable:** the *conditional* statement (3) with `κ` made concrete
  for the Cholesky parameterization, plus empirical sample-complexity curves
  whose slope ratio matches `κ`. (LQR surrogate as the clean analytic anchor.)
- **Stretch:** a distribution-free, two-sided bound (upper for DeLaN, matching
  lower for MLP) establishing a genuine *separation*, in the spirit of
  Tu & Recht (2019) for LQR.

---

## 7. Open questions / TODO

- [ ] Pick the error metric (one-step vs H-step) and propagate consistently.
- [ ] Resolve `√ε` vs `ε` in (1) for that choice.
- [ ] Derive `κ` concretely for the Cholesky mass-matrix net.
- [ ] Write the LQR surrogate where everything is computable; this is the
      fastest path to an advisor sign-off (Phase 1 exit criterion).
- [ ] Handle on-policy (non-i.i.d.) data rigorously or state the assumption.
- [ ] Decide framing: "improved bound" (conditional) vs "separation" (two-sided).
