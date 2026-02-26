import time
import pytest
import numpy as np

from battleship_rl.baselines.heuristic_probmap import HeuristicProbMapAgent

def test_heuristic_no_hang_on_constrained_board():
    """
    Constructs a heavily constrained, nearly impossible board state to
    prove that the Heuristic baseline's internal backtracking search 
    does not hang and properly exits via the step budget.
    """
    agent = HeuristicProbMapAgent(
        board_size=10, 
        ships=[5, 4, 3, 3, 2],
        max_backtrack_steps=200,
    )
    agent.reset()
    
    obs = np.zeros((3, 10, 10), dtype=np.float32)
    
    # 1. Provide a massive checkerboard pattern of misses, destroying contiguous
    # placement options for large ships.
    for r in range(10):
        for c in range(10):
            if (r + c) % 2 == 0:
                obs[1, r, c] = 1.0
                
    agent._update_from_obs(obs, info=None)
    
    # 2. Simulate hitting & sinking a ship of length 2 (ID: 4)
    obs[0, 0, 1] = 1.0
    agent._update_from_obs(obs, info={"outcome_type": "HIT", "outcome_ship_id": 4})
    
    obs[0, 0, 3] = 1.0 # separated bits of hits, physically impossible contiguous placement
    agent._update_from_obs(obs, info={"outcome_type": "SUNK", "outcome_ship_id": 4})
    
    # 3. Add an isolated hit for ship ID 2
    obs[0, 5, 5] = 1.0
    agent._update_from_obs(obs, info={"outcome_type": "HIT", "outcome_ship_id": 2})

    # 4. Trigger the action inference
    start_time = time.time()
    res = agent.act(obs, info=None)
    duration = time.time() - start_time
    
    # Due to the strict 200 constraint budget and the sheer physical impossibility
    # of the checkerboard layout, the backtracking sequence should quickly abort.
    assert duration < 1.0, f"Heuristic agent hung for {duration} seconds! Regression!"
    assert res["fallback_used"] is True, "Heuristic didn't engage fallback despite exhausted budget."
    assert res["fallback_reason"] == "sampling_budget_exhausted_or_zero_prob"

