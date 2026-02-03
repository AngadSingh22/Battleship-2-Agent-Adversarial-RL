#ifndef BATTLESHIP_H
#define BATTLESHIP_H

#include <stdbool.h>
#include <stdint.h>


// Max board size constraints for static allocation (optional)
// We will use dynamic allocation to support arbitrary sizes
// but keeping structs simple.

typedef struct {
  int height;
  int width;
  int32_t *board; // Row-major: -1 = Empty, 0..N-1 = ShipByID
  bool *hits;     // Row-major: true = Hit
  bool *misses;   // Row-major: true = Miss

  int num_ships;
  int *ship_lengths; // Array of lengths
  bool *ship_sunk;   // Array of sunk status

  int steps; // Step counter
} GameState;

// API
#ifdef __cplusplus
extern "C" {
#endif

// Create and destroy
GameState *create_game(int height, int width, int num_ships, int *ship_lengths);
void free_game(GameState *game);

// Core Logic
void reset_game(GameState *game, unsigned int seed); // Random placement
void place_ship_fixed(GameState *game, int ship_id, int r, int c,
                      int orientation); // For deterministic/external placement

// Step
// Returns:
// 0: Miss
// 1: Hit
// 2: Sunk
// -1: Invalid/Repeated
int step_game(GameState *game, int action);

// Observation access
// Fills provided float buffer with (3, H, W) state
void get_observation(GameState *game, float *buffer);

#ifdef __cplusplus
}
#endif

#endif // BATTLESHIP_H
