"""
Quick mask-aware sanity eval for all available Stage 1 checkpoints.
Run as:  PYTHONPATH=. .venv/bin/python3 evaluation/eval_checkpoint.py --regime A
"""
import argparse
import json
import numpy as np
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from battleship_rl.agents.defender import (
    ClusteredDefender, EdgeBiasedDefender, ParityDefender,
    SpreadDefender, UniformRandomDefender,
)
from battleship_rl.envs.battleship_env import BattleshipEnv

DEFENDER_MAP = {
    "UNIFORM":  UniformRandomDefender,
    "EDGE":     EdgeBiasedDefender,
    "CLUSTER":  ClusteredDefender,
    "SPREAD":   SpreadDefender,
    "PARITY":   ParityDefender,
}

def mask_fn(e):
    return e.get_action_mask()


def make_env(cls, seed):
    def _f():
        env = BattleshipEnv(defender=cls(), debug=False)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _f


def eval_model(model, modes, n_episodes, n_envs, seed_offset=8000):
    results = {}
    for name in modes:
        cls = DEFENDER_MAP[name]
        venv = VecMonitor(SubprocVecEnv([
            make_env(cls, seed_offset + i) for i in range(n_envs)
        ]))
        ep_lengths, trunc_count, total_eps = [], 0, 0
        for _ in range(max(1, n_episodes // n_envs)):
            obs = venv.reset()
            dones = [False] * n_envs
            steps = [0]     * n_envs
            while not all(dones):
                masks   = get_action_masks(venv)
                actions, _ = model.predict(obs, deterministic=True, action_masks=masks)
                obs, _, new_dones, infos = venv.step(actions)
                for i, (d, nd) in enumerate(zip(dones, new_dones)):
                    if not d:
                        steps[i] += 1
                        if nd:
                            ep_lengths.append(steps[i])
                            if infos[i].get("TimeLimit.truncated", False):
                                trunc_count += 1
                            total_eps += 1
                            dones[i] = True
        venv.close()
        arr = np.array(ep_lengths)
        fail = trunc_count / max(total_eps, 1)
        results[name] = {
            "mean":      float(arr.mean()),
            "p90":       float(np.percentile(arr, 90)),
            "fail_rate": fail,
        }
        flag = "  ⚠ MASKS STILL BROKEN" if fail > 0.01 else "  ✓"
        print(f"  [{name:7s}] mean={arr.mean():.1f}  p90={np.percentile(arr,90):.1f}"
              f"  fail_rate={fail:.3f}{flag}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime",    default=None, help="A, B, or C (eval all if omitted)")
    parser.add_argument("--n_eps",     type=int, default=100)
    parser.add_argument("--n_envs",    type=int, default=4)
    parser.add_argument("--modes",     nargs="+", default=list(DEFENDER_MAP.keys()))
    parser.add_argument("--checkpoint",default=None, help="Override checkpoint path")
    args = parser.parse_args()

    regimes = [args.regime] if args.regime else ["A", "B", "C"]

    all_results = {}
    for regime in regimes:
        ckpt = Path(args.checkpoint) if args.checkpoint else \
               Path(f"results/training/stage1/{regime}/final_model.zip")
        if not ckpt.exists():
            print(f"[Regime {regime}] Checkpoint not found: {ckpt}  (skipping)")
            continue

        print(f"\n{'='*55}")
        print(f"  Regime {regime} — {args.n_eps} eps/mode  masks=ON")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'='*55}")

        # Load with a dummy env
        dummy_cls = UniformRandomDefender
        dummy_venv = VecMonitor(SubprocVecEnv([make_env(dummy_cls, 0) for _ in range(args.n_envs)]))
        model = MaskablePPO.load(str(ckpt), env=dummy_venv, device="cuda")
        dummy_venv.close()

        results = eval_model(model, args.modes, args.n_eps, args.n_envs)
        all_results[regime] = results

        if "SPREAD" in results and "UNIFORM" in results:
            sp = results["SPREAD"]["mean"]
            un = results["UNIFORM"]["mean"]
            print(f"\n  Δ_gen(SPREAD−UNIFORM) = {sp-un:+.1f}")

        # Write corrected final_eval.json
        out = Path(f"results/training/stage1/{regime}/final_eval_corrected.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"  Saved → {out}")

    # Cross-regime comparison
    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print(f"{'Regime':>8}  {'UNIFORM':>8}  {'SPREAD':>8}  {'Δ':>6}")
        print(f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for regime, res in all_results.items():
            un = res.get("UNIFORM", {}).get("mean", float("nan"))
            sp = res.get("SPREAD",  {}).get("mean", float("nan"))
            print(f"  {regime:>6}    {un:>8.1f}  {sp:>8.1f}  {sp-un:>+6.1f}")


if __name__ == "__main__":
    main()
