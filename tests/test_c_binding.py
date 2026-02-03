import pytest
import ctypes
import numpy as np
from pathlib import Path
from bindings.c_api import CBattleshipFactory, _LIB, GameState

def test_c_struct_layout():
    """Verify struct size and alignment matches assumption. Only if lib loaded."""
    if not _LIB: return
    game = CBattleshipFactory(10, 10, [5])
    assert game.game.contents.height == 10
    assert game.game.contents.width == 10
    assert game.game.contents.num_ships == 1
    assert game.game.contents.ship_lengths[0] == 5
    assert not game.game.contents.hits[0] # Should be false
    assert game.game.contents.steps == 0

def test_c_game_logic():
    """Verify basic rules: Hit, Miss, Sunk."""
    game = CBattleshipFactory(10, 10, [2])
    game.reset(42)
    
    # Place ship manually: Row 0, Col 0, Horizontal
    game.place_ship(0, 0, 0, 0) # ID 0, r=0, c=0, H
    
    # 1. Miss
    res = game.step(10) # (1, 0)
    assert res == 0 # MISS
    
    # 2. Hit (0,0)
    res = game.step(0)
    assert res == 1 # HIT
    
    # 3. Hit (0,1) -> Sunk (Size 2)
    res = game.step(1)
    assert res == 2 # SUNK
    
    # Check Obs
    obs = game.get_obs()
    assert obs.shape == (3, 10, 10)
    assert obs[0, 0, 0] == 1.0 # Hit
    assert obs[0, 0, 1] == 1.0 # Hit
    assert obs[1, 1, 0] == 1.0 # Miss
    assert obs[2, 0, 0] == 0.0 # Unknown cleared
    assert obs[2, 9, 9] == 1.0 # Unknown remains
