# Final Evaluation Suite Results

**Generated:** 2026-02-28 23:35  |  **Episodes per policy:** 100  |  **Seed:** 42

---

## Summary Table

| Policy | Uniform Mean ± Std | Biased Mean ± Std | Δ_gen | Hit Rate | Time-to-First-Hit |
|--------|-------------------|------------------|-------|----------|-------------------|
| Random | 95.5 ± 4.3 | 96.3 ± 4.5 | +0.82 | 0.178 | 4.3 |
| Heuristic | 47.7 ± 9.6 | 46.6 ± 8.7 | -1.13 | 0.357 | 3.6 |
| Particle | 48.4 ± 9.4 | 48.3 ± 9.1 | -0.04 | 0.366 | 3.4 |

---

## Policy: Random
> Random baseline — fires at random valid cells

### Performance Metrics

| Mode | Mean | Std | 90th% | Fail Rate | Fallback Rate |
|------|------|-----|-------|-----------|---------------|
| Uniform | 95.53 | 4.31 | 100.00 | 0.000 | 0.000 |
| Biased  | 96.35 | 4.47 | 100.00 | 0.000 | 0.000 |

**Δ_gen (biased − uniform) = +0.82**

### Behavioural Diagnostics (Uniform Defender, first 30 episodes)

| Metric | Value |
|--------|-------|
| Time to First Hit (shots) | 4.3333 |
| Hit Rate | 0.1779 |
| Hunt Phase Fraction | 0.0453 |
| Revisit Rate (should be 0) | 0.0000 |
| Shots Per Ship Sunk | 19.1133 |

### Interpretation

**Specific:** The random agent requires **95.5** shots on Uniform and **96.3** on Biased, with Δ_gen = 0.82. Hit rate = 0.178, time-to-first-hit = 4.3 shots. This serves as the absolute lower bound; any learned or heuristic agent should do better.

**General:** Random performance establishes the baseline expectation. The hit rate (0.178) is consistent with the geometric probability of hitting a ship cell on a 10×10 board with ships covering ~17 cells out of 100.

---

## Policy: Heuristic
> ProbMap heuristic — probability sampling with constraint backtracking

### Performance Metrics

| Mode | Mean | Std | 90th% | Fail Rate | Fallback Rate |
|------|------|-----|-------|-----------|---------------|
| Uniform | 47.72 | 9.60 | 60.00 | 0.000 | 0.000 |
| Biased  | 46.59 | 8.67 | 60.00 | 0.000 | 0.000 |

**Δ_gen (biased − uniform) = -1.13**

### Behavioural Diagnostics (Uniform Defender, first 30 episodes)

| Metric | Value |
|--------|-------|
| Time to First Hit (shots) | 3.6000 |
| Hit Rate | 0.3566 |
| Hunt Phase Fraction | 0.0755 |
| Revisit Rate (should be 0) | 0.0000 |
| Shots Per Ship Sunk | 9.5333 |

### Interpretation

**Specific:** The ProbMap heuristic requires **47.7** shots on Uniform and **46.6** on Biased, with Δ_gen = -1.13. Hit rate = 0.357, time-to-first-hit = 3.6 shots. Fallback rate = 0.000 (fraction of steps where the sampler exhausted its budget).

**General:** The heuristic substantially outperforms random. The small Δ_gen indicates that the BiasedDefender does not provide a meaningfully harder challenge for a probability-map agent. A gap close to zero or negative would indicate the bias accidentally helps the heuristic (e.g. placing ships on edges where probability mass concentrates naturally).

---

## Policy: Particle
> Particle Belief SMC — full particle filter over board hypotheses

### Performance Metrics

| Mode | Mean | Std | 90th% | Fail Rate | Fallback Rate |
|------|------|-----|-------|-----------|---------------|
| Uniform | 48.38 | 9.44 | 62.00 | 0.000 | 0.000 |
| Biased  | 48.34 | 9.08 | 59.20 | 0.000 | 0.000 |

**Δ_gen (biased − uniform) = -0.04**

### Behavioural Diagnostics (Uniform Defender, first 30 episodes)

| Metric | Value |
|--------|-------|
| Time to First Hit (shots) | 3.3667 |
| Hit Rate | 0.3659 |
| Hunt Phase Fraction | 0.0725 |
| Revisit Rate (should be 0) | 0.0000 |
| Shots Per Ship Sunk | 9.2933 |

### Interpretation

**Specific:** The Particle Belief SMC requires **48.4** shots on Uniform and **48.3** on Biased, with Δ_gen = -0.04. Hit rate = 0.366, time-to-first-hit = 3.4 shots.

**General:** The particle filter is the strongest scripted baseline. It maintains a posterior over full board layouts and marginalizes to select the highest-probability untried cell. A better performance vs heuristic indicates that the explicit constraint propagation (filtering invalid particles vs. re-sampling particles each step) extracts more signal from observations.

---

## Overall Project Interpretation

The results above establish a performance ladder for Battleship attackers:

1. **Random** — 95.5 shots on Uniform
2. **Heuristic** — 47.7 shots on Uniform
3. **Particle** — 48.4 shots on Uniform

The generalisation gap (Δ_gen) measures how much harder the BiasedDefender is vs. Uniform. 
A gap close to 0 means the scripted bias does not present a meaningful challenge; 
a positive gap signals the attacker genuinely struggles against the biased placement strategy.

The next step to produce a larger, more reliable Δ_gen is to train a full `AdversarialDefender` via RL (already implemented in `defender.py` via `AdversarialDefender`), then pass `--adversarial-defender <path>` to the evaluator.