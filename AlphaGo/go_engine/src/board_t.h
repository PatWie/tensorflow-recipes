// Author: Patrick Wieschollek <mail@patwie.com>

#ifndef ENGINE_BOARD_T_H
#define ENGINE_BOARD_T_H

#include "misc.h"
#include "token_t.h"
#include "field_t.h"

#include <array>
#include <memory>
#include <vector>
#include <map>


class board_t {
  public:
    
    /**
     * @brief Create new board representation.
     */
    board_t();
    ~board_t();


    /**
     * @brief Just for convenience.
     * 
     * @param stream board configuration
     * @param b stream
     */
    friend std::ostream& operator<< (std::ostream& stream, const board_t& b);

    /**
     * @brief Set a stone and update groups.
     * 
     * @param x vertical axis (top -> bottom)
     * @param y horizontal axis (left ->right)
     * @param tok color of stone
     * @return 0 if success
     */
    int play(int x, int y, token_t tok);


    int set(int x, int y, token_t tok);

    void update_groups(int x, int y);
    /**
     * @brief Return a group for a given field.
     * @details This creates a new group for the field if the field was without a group.
     * 
     * @param id [description]
     * @return [description]
     */
    group_t* find_or_create_group(int id);

    /**
     * @brief switch perspective of player
     */
    const token_t opponent(token_t tok) const;

    /**
     * @brief test whether placing a token at x, y is legal for given player
     * 
     * @param x [description]
     * @param y [description]
     * @param tok checking for token color of tok
     * @return valid?
     */
    bool is_legal(int x, int y, token_t tok) const ;

    /**
     * @brief place token and count effect of captured stones
     * @details [long description]
     * 
     * @param x [description]
     * @param y [description]
     * @param color_place [description]
     * @param color_count [description]
     * @return number of captured stones
     */
    int estimate_captured_stones(int x, int y, token_t color_place, token_t color_count) const ;

    /**
     * @brief search for groups without liberties and remove them
     * @details [long description]
     * 
     * @param x [description]
     * @param y [description]
     * @param focus just look at groups of this color
     * @return number of removed stones
     */
    int count_and_remove_captured_stones(int x, int y, token_t focus);

    /**
     * @brief compute features of current board configuration as an input for the NN
     * @details ust 47 out of the 49 from the Nature paper
     * 
     * @param planes 47x19x19 values
     * @param self perspective from (predict move for)
     */
    void feature_planes(int *planes, token_t self) const;

    /**
     * @brief count liberties from a field
     * @details could benefit from a caching
     * 
     * @param x [description]
     * @param y [description]
     * 
     * @return [description]
     */
    int liberties(int x, int y) const;

    /**
     * @brief Create a deep copy of current board configuration.
     * @details This copies all properties (fields, groups, counters)
     * @return new board (do not forget to delete this when not used anymore)
     */
    board_t* clone() const ;

    /**
     * @brief get neighboring fields
     * @details hopefully the compiler can optimize this
     * 
     * @param x anchor x
     * @param y anchor y
     * 
     * @return list of (x, y) pairs
     */
     const std::vector<std::pair<int, int> > neighbor_fields(int x, int y)  const;

    bool is_ladder_capture(int x, int y,
                           token_t hunter, token_t current,
                           int recursion_depth, int fx, int fy) const;


    std::array<std::array<field_t, N>, N> fields;
    std::map<int, group_t*> groups;
    std::map<int, group_t*>::iterator groups_iter;

    int groupid;
    int played_moves = 0;
    float score_black;
    float score_white;

};

#endif