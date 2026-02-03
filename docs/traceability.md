# Traceability Checklist

## Encoding Commitments
- [x] **Channel Order**: Hit, Miss, Unknown (`battleship_rl/envs/observations.py`).
- [x] **Action Indexing**: Row-major `a = r * W + c` (`battleship_rl/envs/battleship_env.py`).
- [x] **Sunk-by-ID Semantics**: `sunk_ships` set tracks IDs; `outcome_ship_id` preserves strictly via placement order.
- [x] **Outcome Schema**: `("TYPE", ID/None)` via `info["last_outcome"]`.

## Symbol Mapping

| Symbol | LaTeX Definition | Implementation |
| :--- | :--- | :--- |
| $\mathcal{S}$ | State Space | `BattleshipEnv` internal state: `hits_grid`, `misses_grid`, `ship_status`. |
| $\mathcal{A}$ | Action Space $\{1, \dots, H \times W\}$ | `gym.spaces.Discrete(H*W)` (0-indexed). Mapped in `step()`: `r, c = a // W, a % W`. |
| $\mathcal{O}$ | Observation Space (Grids) | `gym.spaces.Box(3, H, W)`. Channels: `Hit`, `Miss`, `Unknown`. |
| $\tau$ | Trajectory | List of steps stored in `replays` (JSON) or `RolloutBuffer` (SB3). |
| $Z_t$ | Observation Function $Z(s, a, s')$ | `observations.build_observation`. Deterministic mapping from grid state. |
| $U_t$ | Update Function (Belief) | Implicit in `MaskablePPO` (LSTM/CNN state) or explicit in `HeuristicProbMapAgent`. |
| $r_t$ | Reward Function | `StepPenaltyReward` ($r=-1$) or `ShapedReward`. |
| $H, W$ | Grid Dimensions | `BattleshipEnv.height`, `BattleshipEnv.width`. |
| $K$ | Number of Ships | `len(env.ship_lengths)`. |
| $\pi_\text{def}$ | Defender Policy | `UnformRandomDefender` / `BiasedDefender` (`sample_layout`). |
