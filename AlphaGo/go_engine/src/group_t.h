// Author: Patrick Wieschollek <mail@patwie.com>

#ifndef ENGINE_GROUP_T_H
#define ENGINE_GROUP_T_H

#include <vector>
#include <memory>



class field_t;
class board_t;

class group_t {
  public:
    group_t(int groupid);
    ~group_t();

    void add(field_t *s);

    const unsigned int size() const;

    int kill();

    void merge(group_t* other);

    // count liberties (see below)
    // TODO: cache result (key should be iteration in game)
    int liberties(const board_t* const b)  const;

    // collection of pointers to stones
    std::vector<field_t *> stones;
    int id;
};

#endif