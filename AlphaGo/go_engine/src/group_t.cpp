#include <bitset>

#include "misc.h"
// Author: Patrick Wieschollek <mail@patwie.com>

#include "group_t.h"
#include "field_t.h"
#include "board_t.h"




group_t::group_t(int groupid) {
    id = groupid;
}
group_t::~group_t() {}

void group_t::add(field_t *s) {
    // add stone to group
    // check (before calling) if stone "s" belongs to opponent group!
    s->group = this;
    stones.push_back(s);
}

const unsigned int group_t::size() const {
    return stones.size();
}

int group_t::kill() {
    // kill entire group (remove stones from board, destroy group, return score)
    int score = stones.size();
    for (field_t * s : stones) {
        s->token(empty);
        s->played_at = 0;
        s->group = nullptr;
    }
    delete this;
    return score;
}

void group_t::merge(group_t* other) {
    // never merge a group with itself
    if (other->id == id)
        return;

    // merge two groups (current group should own other stones)
    for (field_t * s : other->stones) {
        s->group = this;
        stones.push_back(s);
    }
    // delete other;
}

// to avoid circular dependency
int group_t::liberties(const board_t* const b) const {
    // TODO: this really needs a caching!!!
    // local memory
    std::bitset<19 * 19> already_processed(0);

    for (field_t * s : stones) {

        const int x = s->x();
        const int y = s->y();

        if (valid_pos(y - 1))
            if (!already_processed[map2line(x, y - 1)])
                if (b->fields[x][y - 1].token() == empty)
                    already_processed[map2line(x, y - 1)] = 1;

        if (valid_pos(x - 1))
            if (!already_processed[map2line(x - 1, y)])
                if (b->fields[x - 1][y].token() == empty)
                    already_processed[map2line(x - 1, y)] = 1;

        if (valid_pos(y + 1))
            if (!already_processed[map2line(x, y + 1)])
                if (b->fields[x][y + 1].token() == empty)
                    already_processed[map2line(x, y + 1)] = 1;

        if (valid_pos(x + 1))
            if (!already_processed[map2line(x + 1, y)])
                if (b->fields[x + 1][y].token() == empty)
                    already_processed[map2line(x + 1, y)] = 1;

    }
    return already_processed.count();
}

// collection of pointers to stones
std::vector<field_t *> stones;
int id;
