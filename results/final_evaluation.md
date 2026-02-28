# Final Evaluation Suite Results

**Generated:** 2026-03-01 01:58  |  **Episodes:** 100  |  **Seed:** 42  |  **Workers:** 16

---

## Defender Distribution Metrics

*(500 samples each. `shift_metric` = JSD(q_mode || q_UNIFORM_defender), so UNIFORM=0.000 by construction.)*

| Defender   | EdgeOcc | MeanDist† | Entropy (bits) | JSD vs UNIFORM-defender | Flag |
|------------|---------|-----------|----------------|------------------------|------|
| UNIFORM    |   0.282 |     0.283 |          6.607 |                0.00000 | (reference) |
| EDGE       |   0.464 |     0.190 |          6.588 |                0.03252 | ✓ real shift |
| CLUSTER    |   0.075 |     0.452 |          5.986 |                0.11350 | ✓ real shift |
| SPREAD     |   0.685 |     0.146 |          6.145 |                0.16945 | ✓ real shift |
| PARITY     |   0.276 |     0.285 |          6.546 |                0.01482 | ~ |

†`mean_dist_to_edge` normalized by `floor(min(H,W)/2)` → range [0, 1].

---

## Performance + Diagnostics per Policy

### Policy: Random
> Random baseline — fires at random valid cells

| Defender | Mean±Std | p90 | FailRate | Fallback | Δ_gen | HitRate | T-to-1stHit |
|----------|----------|-----|----------|----------|-------|---------|-------------|
| UNIFORM  | 96.0±4.2 | 100.0 | 0.000 | 0.000 | — | 0.177 | 4.0 |
| EDGE     | 94.7±5.6 | 100.0 | 0.000 | 0.000 | -1.28 | 0.179 | 3.4 |
| CLUSTER  | 95.3±4.8 | 100.0 | 0.000 | 0.000 | -0.68 | 0.178 | 4.7 |
| SPREAD   | 95.5±4.8 | 100.0 | 0.000 | 0.000 | -0.47 | 0.178 | 5.6 |
| PARITY   | 95.5±4.7 | 100.0 | 0.000 | 0.000 | -0.53 | 0.178 | 3.7 |

> **p90 Δ:** Largest tail shift = EDGE (+0.0 vs UNIFORM). ⚠ below p90≥+2 gate

### Policy: Heuristic
> ProbMap heuristic — probability-sampling constraint backtracker

| Defender | Mean±Std | p90 | FailRate | Fallback | Δ_gen | HitRate | T-to-1stHit |
|----------|----------|-----|----------|----------|-------|---------|-------------|
| UNIFORM  | 44.7±8.9 | 58.0 | 0.000 | 0.000 | — | 0.380 | 3.0 |
| EDGE     | 46.5±9.0 | 59.1 | 0.000 | 0.000 | +1.87 | 0.365 | 5.5 |
| CLUSTER  | 46.2±10.6 | 60.0 | 0.000 | 0.000 | +1.51 | 0.368 | 1.4 |
| SPREAD   | 49.5±7.8 | 61.0 | 0.000 | 0.000 | +4.85 | 0.343 | 5.6 |
| PARITY   | 44.5±9.9 | 59.0 | 0.000 | 0.000 | -0.13 | 0.382 | 3.0 |

> **p90 Δ:** Largest tail shift = SPREAD (+3.0 vs UNIFORM). ✓ meets p90≥+2 gate

### Policy: Particle
> Particle Belief SMC — GPU-accelerated full belief filter (P=500)

| Defender | Mean±Std | p90 | FailRate | Fallback | Δ_gen | HitRate | T-to-1stHit |
|----------|----------|-----|----------|----------|-------|---------|-------------|
| UNIFORM  | 48.2±9.3 | 63.1 | 0.000 | 0.000 | — | 0.352 | 3.2 |
| EDGE     | 49.1±10.0 | 64.0 | 0.000 | 0.000 | +0.82 | 0.346 | 5.2 |
| CLUSTER  | 45.7±10.7 | 60.0 | 0.000 | 0.000 | -2.56 | 0.372 | 1.9 |
| SPREAD   | 51.1±7.9 | 63.0 | 0.000 | 0.000 | +2.89 | 0.332 | 5.9 |
| PARITY   | 47.9±9.9 | 62.1 | 0.000 | 0.000 | -0.38 | 0.355 | 3.5 |

> **p90 Δ:** Largest tail shift = EDGE (+0.9 vs UNIFORM). ⚠ below p90≥+2 gate

---

## Sign-off Gate Check

1. **JSD > 0.005 for ≥2 non-UNIFORM modes:** 4/4 modes qualify — ✓ PASS

---

## Diagnosis POMDP Fault-Shift Results

| Baseline | Distribution | Success Rate | Mean Steps (Success) | Δ_gen | Mean Steps (All) |
|----------|-------------|--------------|----------------------|-------|------------------|
| random   | uniform      | 0.090 | 1.56 ± 0.96 | — | 1.64 |
| random   | clustered    | 0.110 | 2.36 ± 1.43 | +0.81 | 1.64 |
| random   | rare_hard    | 0.140 | 1.57 ± 0.62 | +0.02 | 1.64 |
| greedy   | uniform      | 0.760 | 5.83 ± 0.52 | — | 6.04 |
| greedy   | clustered    | 0.840 | 5.96 ± 0.64 | +0.14 | 6.12 |
| greedy   | rare_hard    | 0.830 | 5.59 ± 0.60 | -0.24 | 5.71 |
