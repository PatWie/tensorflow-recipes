// Author: Patrick Wieschollek <mail@patwie.com>

#include <set>
#include <map>
#include <iomanip>

#include "field_t.h"
#include "group_t.h"
#include "board_t.h"


board_t::board_t() : score_black(0.f), score_white(0.f), played_moves(0) {
    // tell each field its coordinate
    for (int h = 0; h < N; ++h)
        for (int w = 0; w < N; ++w)
            fields[h][w].pos(h, w);
    // set counter for group-ids
    groupid = 0;
}

board_t::~board_t() {
    for(auto &g : groups)
        delete g.second;
}



std::ostream& operator<< (std::ostream& stream, const board_t& b) {

    const char *charset = "ABCDEFGHJKLMNOPQRST";

    stream << "   ";
    for (int w = 0; w < N; ++w)
        stream << charset[w] << " ";
    stream << std::endl;
    
    for (int h = N - 1; h >= 0; --h) {
        stream << std::setw(2) << (h + 1)  << " ";
        for (int w = 0; w < N; ++w)
            stream  << b.fields[h][w] << " ";
        stream << std::setw(2) << (h + 1)  << " ";
        stream << std::endl;
    }
    stream << "  ";
    for (int w = 0; w < N; ++w)
        stream << " " << charset[w];
    stream << std::endl;

    return stream;
}


group_t* board_t::find_or_create_group(int id){
    groups_iter = groups.find(id);
    if (groups_iter != groups.end()){
        return groups_iter->second;
    }
    else{
        groups[id] = new group_t(id);
        return groups[id];
    }
}


board_t* board_t::clone() const {

    board_t* dest = new board_t();
    dest->played_moves = played_moves;
    dest->groupid = groupid;

    for (int h = 0; h < N; ++h)
        for (int w = 0; w < N; ++w){
            const field_t &src_f = fields[h][w];
            field_t &dest_f = dest->fields[h][w];

            if (src_f.token() != empty)
            {
                dest_f.token(src_f.token());
                dest_f.played_at = src_f.played_at;

                dest_f.group = dest->find_or_create_group(src_f.group->id);
                dest_f.group->add(&dest_f);
            }
            
        }
    return dest;
}



int board_t::play(int x, int y, token_t tok) {
    int r = set(x, y, tok);
    if (r == -1) return -1;
    return 0;
}

bool board_t::is_ladder_capture(int x, int y,
                                token_t hunter, token_t current,
                                int recursion_depth, int fx, int fy) const{
    // does placing a token at (x,y) from hunter would capture a group?
    // we focus on the group in fx, fy
    const token_t victim = opponent(hunter);

    if(!is_legal(x,y, hunter)){
        // nope not here
        return false;
    }
    if(recursion_depth == 0){
        // haven't found anything, so this might not be a ladder capture
        return false;
    }

    std::vector<std::pair<int, int> > possible_group_victims;

    if((fx==-1) && (fy==-1)){
        // not a particular group focus
        const auto neighbors = neighbor_fields(x, y);
        for(auto &&n : neighbors){
            const field_t& other_stone = fields[n.first][n.second];
            if(other_stone.token() == victim){
                // we could capture that stone
                if(liberties(n.first, n.second) == 2){
                    // it is more likely that we capture that stone
                    possible_group_victims.push_back({n.first, n.second});
                }
            }
        }
    }else{
        // only focus on group of (fx, fy)
        possible_group_victims.push_back({fx, fy});
    }

    for(auto &&pos : possible_group_victims){
        board_t* copy = clone();
        copy->play(x, y, current);

        // TODO ....

        delete copy;
    }



}


int board_t::set(int x, int y, token_t tok) {
    if (!valid_pos(x) || !valid_pos(y)) {
        std::cout << "field is not valid" << std::endl;
        return -1;
    }

    if (fields[x][y].token() != empty) {
        std::cout << "field was not empty" << std::endl;
        return -1;
    }

    if(!is_legal(x, y, tok)){
        std::cout << "move is not legal" << std::endl;
        return -1;
    }

    // place token to field
    fields[x][y].token(tok);
    fields[x][y].played_at = played_moves++;

    // update group structures
    update_groups(x, y);

    // does this move captures some opponent stones?
    int taken = count_and_remove_captured_stones(x, y, opponent(tok));

    // TODO: check suicidal move, i.e. a move which kills its own group
    // TODO: check ko, i.e. the same position should not appear on the board again (hashing?)

    // maintain scores
    if (taken > 0) {
        if (tok == white)
            score_white += taken;
        if (tok == black)
            score_black += taken;
    }
    return 0;
}

 const std::vector<std::pair<int, int> > board_t::neighbor_fields(int x, int y) const {
    std::vector<std::pair<int, int> > n;
    if(valid_pos(x - 1))
        n.push_back({x - 1, y});
    if(valid_pos(x + 1))
        n.push_back({x + 1, y});
    if(valid_pos(y - 1))
        n.push_back({x, y - 1});
    if(valid_pos(y + 1))
        n.push_back({x, y + 1});
    return n;
}

/**
 * @brief Update groups (merge, kill)
 * @details Update all neighboring groups (add current field and merge groups)
 * 
 * @param x focused token position
 * @param y focused token position
 */
void board_t::update_groups(int x, int y) {

    token_t current = fields[x][y].token();
    field_t& current_stone = fields[x][y];

    const auto neighbors = neighbor_fields(x, y);
    for(auto &&n : neighbors){
        field_t& other_stone = fields[n.first][n.second];
        if (other_stone.token() == current) {
            if (current_stone.group == nullptr)
                other_stone.group->add(&current_stone);
            else
                current_stone.group->merge(other_stone.group);
        }
    }

    // still single stone ? --> create new group
    if (current_stone.group == nullptr) {
        current_stone.group = find_or_create_group(groupid++);
        current_stone.group->add(&current_stone);
    }

}

const token_t board_t::opponent(token_t tok) const {
    return (tok == white) ? black : white;
}

bool board_t::is_legal(int x, int y, token_t tok) const {
    // position has already a stone ?
    if (fields[x][y].token() != empty)
        return false;

    // // check own eye (special case from below)
    // bool is_eye = true;
    // if (is_eye && valid_pos(x + 1))
    //     if(fields[x + 1][y].token() != tok)
    //         is_eye = false;
    // if (is_eye && valid_pos(x - 1))
    //     if(fields[x - 1][y].token() != tok)
    //         is_eye = false;
    // if (is_eye && valid_pos(y + 1))
    //     if(fields[x][y + 1].token() != tok)
    //         is_eye = false;
    // if (is_eye && valid_pos(y - 1))
    //     if(fields[x][y - 1].token() != tok)
    //         is_eye = false;

    // if(is_eye)
    //     return false;

    // test self atari
    // placing that stone would cause the group to only have liberty afterwards
    board_t* copy = clone();
    copy->fields[x][y].token(tok);
    copy->update_groups(x, y);
    copy->count_and_remove_captured_stones(x, y, opponent(tok));
    copy->count_and_remove_captured_stones(x, y, tok);
    bool self_atari = (copy->liberties(x, y) == 0);
    // bool self_atari = (copy->fields[x][y].group->liberties(copy) == 0);
    delete copy;
    return !self_atari;

}

int board_t::estimate_captured_stones(int x, int y, token_t color_place, token_t color_count)  const {

    if (fields[x][y].token() != empty)
        return 0;

    board_t* copy = clone();
    copy->fields[x][y].token(color_place);
    copy->update_groups(x, y);
    int scores = copy->count_and_remove_captured_stones(x, y, color_count);
    delete copy;

    return scores;
}

int board_t::count_and_remove_captured_stones(int x, int y, token_t focus) {
    int scores = 0;

    const auto neighbors = neighbor_fields(x, y);
    for(auto &&n : neighbors){
        field_t& other_stone = fields[n.first][n.second];
        if (other_stone.token() == focus)
            if (liberties(n.first, n.second) == 0){
                const int gid = other_stone.group->id;
                scores += other_stone.group->kill();
                groups.erase(gid);
            }
    }

    return scores;
}

// int board_t::liberties(const field_t field) const{
int board_t::liberties(int x, int y) const{
    if(fields[x][y].token() == empty)
        return 0;
    else
        return fields[x][y].group->liberties(this);

}

void board_t::feature_planes(int *planes, token_t self) const {
    // see https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf (Table 2)

    const int NN = 19 * 19;
    const token_t other = opponent(self);

    for (int h = 0; h < N; ++h) {
        for (int w = 0; w < N; ++w) {

            // Stone colour 3
            // 1x mark all fields with own tokens
            // 1x mark all fields with opponent tokens
            // 1x mark all empty fields
            if (fields[h][w].token() == self)
                planes[map3line(0, h, w)] = 1;
            else if (fields[h][w].token() == other)
                planes[map3line(1, h, w)] = 1;
            else
                planes[map3line(2, h, w)] = 1;

            // Ones
            // fill entire plane with ones
            planes[map3line(3, h, w)] = 1;

            // Turns since
            // counter number of turns since the token was placed
            if (fields[h][w].token() != empty) {

                const int since = played_moves - fields[h][w].played_at + 1;

                if (since == 1)
                    planes[map3line(4, h, w)] = 1;
                else if (since == 2)
                    planes[map3line(5, h, w)] = 1;
                else if (since == 3)
                    planes[map3line(6, h, w)] = 1;
                else if (since == 4)
                    planes[map3line(7, h, w)] = 1;
                else if (since == 5)
                    planes[map3line(8, h, w)] = 1;
                else if (since == 6)
                    planes[map3line(9, h, w)] = 1;
                else if (since == 7)
                    planes[map3line(10, h, w)] = 1;
                else if (since > 7)
                    planes[map3line(11, h, w)] = 1;
            }

            // Liberties
            // 8x count number of liberties of own groups
            if (fields[h][w].token() == self) {

                const int num_liberties = liberties(h, w);

                if (num_liberties == 1)
                    planes[map3line(12, h, w)] = 1;
                else if (num_liberties == 2)
                    planes[map3line(13, h, w)] = 1;
                else if (num_liberties == 3)
                    planes[map3line(14, h, w)] = 1;
                else if (num_liberties == 4)
                    planes[map3line(15, h, w)] = 1;
                else if (num_liberties == 5)
                    planes[map3line(16, h, w)] = 1;
                else if (num_liberties == 6)
                    planes[map3line(17, h, w)] = 1;
                else if (num_liberties == 7)
                    planes[map3line(18, h, w)] = 1;
                else if (num_liberties > 7)
                    planes[map3line(19, h, w)] = 1;
            }

            // Liberties
            // 8x count number of liberties of opponent groups
            if (fields[h][w].token() == other) {

                const int num_liberties = liberties(h, w);

                if (num_liberties == 1)
                    planes[map3line(20, h, w)] = 1;
                else if (num_liberties == 2)
                    planes[map3line(21, h, w)] = 1;
                else if (num_liberties == 3)
                    planes[map3line(22, h, w)] = 1;
                else if (num_liberties == 4)
                    planes[map3line(23, h, w)] = 1;
                else if (num_liberties == 5)
                    planes[map3line(24, h, w)] = 1;
                else if (num_liberties == 6)
                    planes[map3line(25, h, w)] = 1;
                else if (num_liberties == 7)
                    planes[map3line(26, h, w)] = 1;
                else if (num_liberties > 7)
                    planes[map3line(27, h, w)] = 1;
            }

            // Capture size
            // 8x How many opponent stones would be captured when playing this field?
            if (fields[h][w].token() == empty) {

                const int num_capture = estimate_captured_stones(h, w, self, other);

                if (num_capture == 1)
                    planes[map3line(28, h, w)] = 1;
                else if (num_capture == 2)
                    planes[map3line(29, h, w)] = 1;
                else if (num_capture == 3)
                    planes[map3line(30, h, w)] = 1;
                else if (num_capture == 4)
                    planes[map3line(31, h, w)] = 1;
                else if (num_capture == 5)
                    planes[map3line(32, h, w)] = 1;
                else if (num_capture == 6)
                    planes[map3line(33, h, w)] = 1;
                else if (num_capture == 7)
                    planes[map3line(34, h, w)] = 1;
                else if (num_capture > 7)
                    planes[map3line(35, h, w)] = 1;
            }

            // Self-atari size
            // 8x How many own stones would be captured when playing this field?
            if (fields[h][w].token() == empty) {

                const int num_capture = estimate_captured_stones(h, w, self, self);

                if (num_capture == 1)
                    planes[map3line(36, h, w)] = 1;
                else if (num_capture == 2)
                    planes[map3line(37, h, w)] = 1;
                else if (num_capture == 3)
                    planes[map3line(38, h, w)] = 1;
                else if (num_capture == 4)
                    planes[map3line(39, h, w)] = 1;
                else if (num_capture == 5)
                    planes[map3line(40, h, w)] = 1;
                else if (num_capture == 6)
                    planes[map3line(41, h, w)] = 1;
                else if (num_capture == 7)
                    planes[map3line(42, h, w)] = 1;
                else if (num_capture > 7)
                    planes[map3line(43, h, w)] = 1;
            }

            // TODO Ladder c'mon (not here)
            // Ladder capture : 1 : Whether a move at this point is a successful ladder capture
            // Ladder escape : 1 : Whether a move at this point is a successful ladder escape
            // these two features are probably the most important ones, as they allow to look into the future
            // ... but they are missing here ;-)
            // this would require a small recursion

            // Sensibleness : 1 : Whether a move is legal and does not fill its own eyes
            if (!is_legal(h, w, self)) {
                planes[map3line(44, h, w)] = 1;
            }

            // Zeros : 1 : A constant plane filled with 0
            planes[map3line(45, h, w)] = 0;

            // Player color :1: Whether current player is black
            const int value = (self == black) ? 1 : 0;
            planes[map3line(46, h, w)] = value;

        }
    }
}

