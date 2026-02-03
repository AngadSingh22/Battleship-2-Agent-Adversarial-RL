import numpy as np

from battleship_rl.envs.masks import compute_action_mask


def test_action_mask_excludes_hits_and_misses():
    hits = np.zeros((3, 3), dtype=bool)
    misses = np.zeros((3, 3), dtype=bool)
    hits[0, 0] = True
    misses[1, 1] = True

    mask = compute_action_mask(hits, misses)
    assert mask.shape == (9,)
    assert mask[0] == False
    assert mask[4] == False
    assert mask.sum() == 7
