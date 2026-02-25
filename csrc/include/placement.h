#ifndef PLACEMENT_H
#define PLACEMENT_H

#include "battleship.h" /* GameState */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Randomly place all ships in `game` using the provided seed.
 * Ships are placed one at a time (in index order) into a random legal cell.
 * Returns 1 on success, 0 if any ship cannot be placed (board too full).
 *
 * The caller must have already called reset_game() (without placement) so
 * that game->board is fully initialised to -1 before calling this function.
 */
int place_ships_random(GameState *game, unsigned int seed);

/*
 * Validate a proposed (ship_id, r, c, orientation) placement.
 * Returns 1 if the placement is legal (in bounds, no overlap), 0 otherwise.
 */
int is_valid_placement(const GameState *game, int ship_id, int r, int c,
                       int orientation);

#ifdef __cplusplus
}
#endif

#endif /* PLACEMENT_H */
