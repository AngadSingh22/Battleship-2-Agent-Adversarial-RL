import ctypes
import sys
from pathlib import Path
import numpy as np

# Resolve platform library path: .so on Linux/macOS, .dll on Windows.
_CSRC_DIR = Path(__file__).parent.parent / "csrc"
if sys.platform.startswith("win"):
    _LIB_PATH = _CSRC_DIR / "libbattleship_v2.dll"
else:
    _LIB_PATH = _CSRC_DIR / "libbattleship_v2.so"

try:
    _LIB = ctypes.CDLL(str(_LIB_PATH))
except OSError as e:
    print(f"Warning: Could not load C library from {_LIB_PATH}: {e}")
    print("Falling back to PyBattleship (pure Python). Run `make` to build the C extension.")
    _LIB = None

# Struct Definition
class GameState(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("board", ctypes.POINTER(ctypes.c_int32)),
        ("hits", ctypes.POINTER(ctypes.c_bool)),
        ("misses", ctypes.POINTER(ctypes.c_bool)),
        ("num_ships", ctypes.c_int),
        ("ship_lengths", ctypes.POINTER(ctypes.c_int)),
        ("ship_sunk", ctypes.POINTER(ctypes.c_bool)),
        ("steps", ctypes.c_int),
    ]

# Signatures
if _LIB:
    # GameState* create_game(int height, int width, int num_ships, int* ship_lengths);
    _LIB.create_game.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    _LIB.create_game.restype = ctypes.POINTER(GameState)

    # void free_game(GameState* game);
    _LIB.free_game.argtypes = [ctypes.POINTER(GameState)]
    _LIB.free_game.restype = None

    # void reset_game(GameState* game, unsigned int seed);
    _LIB.reset_game.argtypes = [ctypes.POINTER(GameState), ctypes.c_uint]
    _LIB.reset_game.restype = None

    # void place_ship_fixed(GameState* game, int ship_id, int r, int c, int orientation);
    _LIB.place_ship_fixed.argtypes = [ctypes.POINTER(GameState), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _LIB.place_ship_fixed.restype = None

    # int step_game(GameState* game, int action);
    _LIB.step_game.argtypes = [ctypes.POINTER(GameState), ctypes.c_int]
    _LIB.step_game.restype = ctypes.c_int

    # void get_observation(GameState* game, float* buffer);
    _LIB.get_observation.argtypes = [ctypes.POINTER(GameState), ctypes.POINTER(ctypes.c_float)]
    _LIB.get_observation.restype = None

# Helpers for Python
class CBattleship:
    def __init__(self, height=10, width=10, ships=[5, 4, 3, 3, 2]):
        if not _LIB:
            raise RuntimeError("C Library not loaded")
        
        self.height = height
        self.width = width
        self.ships = np.array(ships, dtype=np.int32)
        
        c_ships = self.ships.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.game = _LIB.create_game(height, width, len(ships), c_ships)
        
        # Buffer for observations (pre-allocated) — 4 channels: ActiveHit, Miss, Sunk, Unknown
        self.obs_buffer = np.zeros((4, height, width), dtype=np.float32)

        # Create Numpy Views of C-Memory (Zero Copy)
        self.board = np.ctypeslib.as_array(self.game.contents.board, shape=(height, width))
        self.hits = np.ctypeslib.as_array(self.game.contents.hits, shape=(height, width))
        self.misses = np.ctypeslib.as_array(self.game.contents.misses, shape=(height, width))
        sunk_ptr = self.game.contents.ship_sunk
        self.ship_sunk = np.ctypeslib.as_array(sunk_ptr, shape=(len(ships),))


    def __del__(self):
        if _LIB and self.game:
            _LIB.free_game(self.game)

    def reset(self, seed=0):
        _LIB.reset_game(self.game, seed)

    def set_board(self, board_grid: np.ndarray):
        """Memcpy python grid to C board."""
        if board_grid.shape != (self.height, self.width):
             raise ValueError("Shape mismatch")
        # Ensure contiguous int32
        src = np.ascontiguousarray(board_grid, dtype=np.int32)
        ctypes.memmove(self.game.contents.board, src.ctypes.data, src.nbytes)

    def place_ship(self, ship_id, r, c, orientation):
        _LIB.place_ship_fixed(self.game, ship_id, r, c, orientation)

    def step(self, action):
        return _LIB.step_game(self.game, action)

    def get_obs(self):
        c_buf = self.obs_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _LIB.get_observation(self.game, c_buf)
        return self.obs_buffer

class PyBattleship:
    """Pure Python fallback for C API. Matches CBattleship interface."""
    def __init__(self, height=10, width=10, ships=[5, 4, 3, 3, 2]):
        self.height = height
        self.width = width
        self.ships = list(ships)
        # Mimic inner C struct pointers with Python lists/arrays if needed, 
        # but here we just implement logic.
        self.game = self # To satisfy .game access check if any
        
        self.board = np.full((height, width), -1, dtype=np.int32)
        self.hits = np.zeros((height, width), dtype=bool)
        self.misses = np.zeros((height, width), dtype=bool)
        self.ship_sunk = [False] * len(ships)
        self.steps = 0
        
    def reset(self, seed=0):
        self.board.fill(-1)
        self.hits.fill(False)
        self.misses.fill(False)
        self.ship_sunk = [False] * len(self.ships)
        self.steps = 0

    def set_board(self, board_grid: np.ndarray):
        """Mimic memcpy."""
        if board_grid.shape != (self.height, self.width):
             raise ValueError("Shape mismatch")
        self.board[:] = board_grid[:]
        
    def place_ship(self, ship_id, r, c, orientation):
        length = self.ships[ship_id]
        if orientation == 0: # Horizontal
            self.board[r, c : c + length] = ship_id
        else: # Vertical
            self.board[r : r + length, c] = ship_id
            
    def step(self, action):
        r, c = divmod(action, self.width)
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return -1
            
        if self.hits[r, c] or self.misses[r, c]:
            return -1
            
        self.steps += 1
        ship_id = self.board[r, c]
        
        if ship_id != -1:
            self.hits[r, c] = True
            # Sunk Check
            # Find all cells for this ship
            coords = np.argwhere(self.board == ship_id)
            is_sunk = np.all(self.hits[coords[:, 0], coords[:, 1]])
            if is_sunk:
                self.ship_sunk[ship_id] = True
                return 2 # SUNK
            return 1 # HIT
        else:
            self.misses[r, c] = True
            return 0 # MISS

    def get_obs(self):
        """Return 4-channel obs: ActiveHit, Miss, Sunk, Unknown."""
        obs = np.zeros((4, self.height, self.width), dtype=np.float32)
        hits = self.hits.astype(bool)
        misses = self.misses.astype(bool)
        # Sunk mask: cells belonging to a sunk ship that have been hit
        sunk_mask = np.zeros((self.height, self.width), dtype=bool)
        for ship_id, sunk in enumerate(self.ship_sunk):
            if sunk:
                sunk_mask |= (self.board == ship_id)
        obs[0] = (hits & ~sunk_mask).astype(np.float32)  # ActiveHit
        obs[1] = misses.astype(np.float32)                # Miss
        obs[2] = sunk_mask.astype(np.float32)             # Sunk
        obs[3] = (~hits & ~misses).astype(np.float32)     # Unknown
        return obs

# Factory to choose backend
def CBattleshipFactory(height=10, width=10, ships=[5, 4, 3, 3, 2]):
    if _LIB:
        return CBattleship(height, width, ships)
    else:
        # print("Warning: Using Python fallback for C kernel.")
        return PyBattleship(height, width, ships)

