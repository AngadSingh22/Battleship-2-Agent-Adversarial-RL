#include "../include/battleship.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Memory Management ---

BATTLESHIP_API GameState *create_game(int height, int width, int num_ships,
                                      int *ship_lengths) {
  GameState *game = (GameState *)malloc(sizeof(GameState));
  if (!game)
    return NULL;

  game->height = height;
  game->width = width;
  game->num_ships = num_ships;

  // Grids
  game->board = (int32_t *)malloc(height * width * sizeof(int32_t));
  game->hits = (bool *)malloc(height * width * sizeof(bool));
  game->misses = (bool *)malloc(height * width * sizeof(bool));

  // Ship Metadata
  game->ship_lengths = (int *)malloc(num_ships * sizeof(int));
  memcpy(game->ship_lengths, ship_lengths, num_ships * sizeof(int));

  game->ship_sunk = (bool *)malloc(num_ships * sizeof(bool));

  return game;
}

BATTLESHIP_API void free_game(GameState *game) {
  if (game) {
    if (game->board)
      free(game->board);
    if (game->hits)
      free(game->hits);
    if (game->misses)
      free(game->misses);
    if (game->ship_lengths)
      free(game->ship_lengths);
    if (game->ship_sunk)
      free(game->ship_sunk);
    free(game);
  }
}

// --- Logic ---

BATTLESHIP_API void reset_game(GameState *game, unsigned int seed) {
  // 1. Reset grids
  // -1 = Empty
  for (int i = 0; i < game->height * game->width; i++) {
    game->board[i] = -1;
    game->hits[i] = false;
    game->misses[i] = false;
  }

  // 2. Reset ship status
  for (int i = 0; i < game->num_ships; i++) {
    game->ship_sunk[i] = false;
  }

  game->steps = 0;

  // Seed usage: In Phase 3 pure C, we can implement random placement.
  // For now, exposing "place_ship_fixed" so Python can drive placement logic
  // or we add a simple C-based random placement here if desired.
  // For completeness with "random" contract:
  srand(seed);
}

BATTLESHIP_API void place_ship_fixed(GameState *game, int ship_id, int r, int c,
                                     int orientation) {
  // No validation here, assumes caller is valid!
  int len = game->ship_lengths[ship_id];
  if (orientation == 0) { // Horizontal
    for (int k = 0; k < len; k++) {
      game->board[r * game->width + (c + k)] = ship_id;
    }
  } else { // Vertical
    for (int k = 0; k < len; k++) {
      game->board[(r + k) * game->width + c] = ship_id;
    }
  }
}

static bool check_sunk(GameState *game, int ship_id) {
  // Scan board for this ship_id, verify all are hit
  bool all_hit = true;
  for (int i = 0; i < game->height * game->width; i++) {
    if (game->board[i] == ship_id) {
      if (!game->hits[i]) {
        all_hit = false;
        break;
      }
    }
  }
  return all_hit;
}

BATTLESHIP_API int step_game(GameState *game, int action) {
  if (action < 0 || action >= game->height * game->width) {
    return -1; // Invalid range
  }

  // Check repeatability
  if (game->hits[action] || game->misses[action]) {
    return -1; // Repeated
  }

  int ship_id = game->board[action];
  game->steps++;

  if (ship_id != -1) {
    // HIT
    game->hits[action] = true;

    // Sunk Check?
    if (check_sunk(game, ship_id)) {
      game->ship_sunk[ship_id] = true;
      return 2; // SUNK
    }
    return 1; // HIT
  } else {
    // MISS
    game->misses[action] = true;
    return 0; // MISS
  }
}

BATTLESHIP_API void get_observation(GameState *game, float *buffer) {
  // Fill buffer (4 * H * W)
  // Channel 0: Active Hits (Hit but ship not yet sunk)
  // Channel 1: Misses
  // Channel 2: Sunk (Hit and ship is sunk)
  // Channel 3: Unknown (Not visited)

  int size = game->height * game->width;
  for (int i = 0; i < size; i++) {
    bool is_hit = game->hits[i];
    bool is_miss = game->misses[i];
    int ship_id = game->board[i];
    bool is_sunk = (ship_id != -1) && game->ship_sunk[ship_id];

    // Mutually exclusive encoding
    float val_active_hit = (is_hit && !is_sunk) ? 1.0f : 0.0f;
    float val_miss = is_miss ? 1.0f : 0.0f;
    float val_sunk = is_sunk ? 1.0f : 0.0f;
    // Unknown if neither hit nor miss
    float val_unknown = (!is_hit && !is_miss) ? 1.0f : 0.0f;

    buffer[0 * size + i] = val_active_hit;
    buffer[1 * size + i] = val_miss;
    buffer[2 * size + i] = val_sunk;
    buffer[3 * size + i] = val_unknown;
  }
}
