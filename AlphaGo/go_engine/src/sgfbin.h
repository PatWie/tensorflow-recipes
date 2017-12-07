// Author: Patrick Wieschollek <mail@patwie.com>

#ifndef ENGINE_SGFBIN_H
#define ENGINE_SGFBIN_H

#include <iostream>
#include <vector>
#include <string>

/**
 * @brief Reader for binary SGF files
 * @details It seems to be easier to load a binary version of GO-specific SGF files
 *          which are converted from ASCII SGF files by a python script.
 */
class SGFbin {
  public:
    /**
     * @brief load binary SGF file given filename
     *
     * @param path path to binary file
     */
    SGFbin(std::string path);

    /**
     * @brief load binary SGF from a buffer
     * @details This is mostly for SWIG+Python bindings, where python serves np.arrays(np.uint8)
     *
     * @param char buffer containing the moves from binary SGFfile
     * @param len length of buffer
     */
    SGFbin(unsigned char* buffer, int len);

    /**
     * @brief read move from SGFbin description
     *
     * @param x position [1-19]
     * @param y position [A-S]
     * @param is_white move was from player "white"
     * @param is_move move was really "playing a move" and not a "set stone" (see SGF format for details)
     * @param is_pass player passed this move
     */
    void parse(unsigned int step,
                      int *x, int *y, 
                      bool *is_white, bool *is_move, bool *is_pass);

    void debug(unsigned int step);

    /**
     * @brief decodes a move from 2 bytes
     * @details Each move is represented by 2 bytes which can be divided into
     *    format:       ---pmcyy|yyyxxxxx (low bit)
     *    p: is passed ? [1:yes, 0:no]
     *    m: is move ? [1:yes, 0:no] (sgf supports 'set' as well)
     *    c: is white ? [1:yes, 0:no] (0 means black ;-) )
     *    y: encoded row (1-19)
     *    x: encoded column (a-s)
     *    -: free bits (maybe we can store something useful here, current sun position or illuminati-code?)
     */
    void parse(unsigned char m1, unsigned char m2,
                        int *x, int *y, 
                        bool *is_white, bool *is_move, bool *is_pass);

    /**
     * @brief number of total actions for game (including "set")
     */
    const unsigned int num_actions() const;

    /**
     * @brief reconstruct plain ASCII version of SGF file (only important things)
     */
    void ascii();

  private:
    /**
     * @brief reading from file o buffer
     * @param filename path to SGFbin file
     */
    std::vector<char> read_moves(char const* filename);
    std::vector<char> moves_;
};

#endif