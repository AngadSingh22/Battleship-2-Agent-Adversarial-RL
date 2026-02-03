from battleship_rl.envs.battleship_env import BattleshipEnv


def test_action_mapping_bijection():
    env = BattleshipEnv(board_size=(3, 4), ships=[2])
    height, width = env.height, env.width
    assert env.action_space.n == height * width

    for r in range(height):
        for c in range(width):
            action = r * width + c
            assert action // width == r
            assert action % width == c
