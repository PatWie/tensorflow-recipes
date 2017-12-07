// Author: Patrick Wieschollek <mail@patwie.com>

#include "../src/token_t.h"
#include "../src/field_t.h"
#include "../src/group_t.h"
#include "../src/board_t.h"
#include "../src/sgfbin.h"
#include "goplanes.h"




int play_game(SGFbin *Game, int* data, const int moves) {

    // board representation
    board_t b;

    // properties of ply
    token_t opponent_player, current_player;
    int x = 0, y = 0;
    bool is_white = true, is_move = true, is_pass = true;

    // for (int i = 0; i < 10; ++i)
    //     Game->debug(i);

    int offset = 0;
    Game->parse(offset, &x, &y, &is_white, &is_move, &is_pass);

    // place all handicap stones
    while (!is_move && !is_pass) {
        // printf("%i: set handicap stone\n", offset);
        offset++;
        current_player = is_white ? white : black;
        opponent_player = is_white ? black : white;
        b.set(x, y, current_player);
        Game->parse(offset, &x, &y, &is_white, &is_move, &is_pass);
    }


    // run game for at least 'moves' moves but stop early enough such that a last move remains open
    int evaluate_until = std::min(offset + moves, (int)Game->num_actions() - 1);

    // really all moves?
    if(moves == -1)
        evaluate_until = Game->num_actions();

    for (; offset < evaluate_until; offset++) {
        // parse move
        Game->parse(offset, &x, &y, &is_white, &is_move, &is_pass);
        // Game->debug(offset);

        current_player = is_white ? white : black;
        opponent_player = is_white ? black : white;

        if (!is_pass) {
            if (is_move) {
                b.play(x, y, current_player);
            } else {
                b.set(x, y, current_player);
            }
        }
    }

    // given the current situation, we switch to the view of the opponent (the play who's turn it is)
    b.feature_planes(data, opponent_player);

    // all moves are evaluated nothing to do
    if(moves == -1)
        return 0;
    // parse a next move (the ground-truth)
    Game->parse(evaluate_until, &x, &y, &is_white, &is_move, &is_pass);
    const int next_move = 19 * y + x;
    return next_move;
}


/**
 * @brief return board configuration and next move given a file
 * @details SWIG-Python-binding
 *
 * @param str path to file
 * @param strlen length of that path name
 * @param data pointer of features (length 19*19*14) for 14 feature planes
 * @param len currently 19*19*4
 * @param moves number of moves in match to the current position
 * @return next move on board
 */
int planes_from_file(char *str, int strlen,
                     int* data, int dc, int dh, int dw,
                     int moves) {
    // load game
    std::string path = std::string(str);
    SGFbin Game(path);
    return play_game(&Game, data, moves);
}




/**
 * @brief return board configuration and next move given a file
 * @details SWIG-Python-binding
 *
 * paper: "... Each position consisted of a raw board description s
 *         and the move a selected by the human. ..."
 *
 * @param bytes buffer of moves (each move 2 bytes)
 * @param strlen length of buffer
 * @param data pointer of features (length 19*19*14) for 14 feature planes (this is "s")
 * @param len currently 19*19*4
 * @param moves number of moves in match to the current position
 * @return next move on board (this is "a")
 */
int planes_from_bytes(char *bytes, int byteslen, int* data, int dc, int dh, int dw, int moves) {
    // the SGFbin parser
    SGFbin Game((unsigned char*) bytes, byteslen);
    return play_game(&Game, data, moves);
}


/**
 * @brief return board configuration and next move given a board position
 * @details SWIG-Python-binding
 */
void planes_from_position(int* bwhite, int wm, int wn,
                          int* bblack, int bm, int bn,
                          int* data, int dc, int dh, int dw,
                          int is_white) {

    board_t b;

    for (int x = 0; x < 19; ++x)
    {
        for (int y = 0; y < 19; ++y)
        {
            if (bwhite[19 * x + y] == 1) {
                b.play(x, y, white);
            }
            if (bblack[19 * x + y] == 1) {
                b.play(x, y, black);
            }
        }
    }

    // std::cout << "----------------- TFGO --------------------" << std::endl;
    // std::cout << b << std::endl;
    // std::cout << "-------------------------------------------" << std::endl;

    // create board configuration from perspective of 'tok'
    token_t tok = (is_white == 1) ? white : black;
    b.feature_planes(data, tok);
}

