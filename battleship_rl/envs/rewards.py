from __future__ import annotations


class StepPenaltyReward:
    """Step penalty reward aligned with shots-to-win."""

    def __init__(self, step_penalty: float = -1.0, terminal_reward: float = 0.0):
        self.step_penalty = float(step_penalty)
        self.terminal_reward = float(terminal_reward)

    def __call__(self, outcome_type: str, terminated: bool) -> float:
        if terminated:
            return self.terminal_reward
        return self.step_penalty


class ShapedReward:
    """Shaped reward: -1 + alpha * hit + beta * sunk."""

    def __init__(self, alpha: float = 0.5, beta: float = 1.0, step_penalty: float = -1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.step_penalty = float(step_penalty)

    def __call__(self, outcome_type: str, terminated: bool) -> float:
        reward = self.step_penalty
        if outcome_type == "HIT":
            reward += self.alpha
        elif outcome_type == "SUNK":
            reward += self.beta
        return reward
