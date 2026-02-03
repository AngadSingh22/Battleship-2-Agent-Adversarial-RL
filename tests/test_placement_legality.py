import numpy as np

from battleship_rl.envs.placement import sample_placement


def _is_contiguous(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        sorted_cols = np.sort(cols)
        return sorted_cols[-1] - sorted_cols[0] + 1 == len(sorted_cols)
    if np.all(cols == cols[0]):
        sorted_rows = np.sort(rows)
        return sorted_rows[-1] - sorted_rows[0] + 1 == len(sorted_rows)
    return False


def test_sample_placement_legality():
    rng = np.random.default_rng(123)
    board = sample_placement(5, [3, 2], rng)
    assert board.shape == (5, 5)

    for ship_id, length in enumerate([3, 2]):
        coords = np.argwhere(board == ship_id)
        assert coords.shape[0] == length
        assert _is_contiguous(coords)

    assert np.all((board == -1) | (board >= 0))
