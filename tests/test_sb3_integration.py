from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from battleship_rl.agents.policies import BattleshipCnnPolicy
from battleship_rl.envs.battleship_env import BattleshipEnv


def test_sb3_integration_smoke():
    def make_env():
        env = BattleshipEnv(board_size=4, ships=[2])
        return ActionMasker(env, lambda e: e.get_action_mask())

    vec_env = DummyVecEnv([make_env])
    model = MaskablePPO(
        BattleshipCnnPolicy,
        vec_env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
    )
    model.learn(total_timesteps=32)
