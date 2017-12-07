// Author: Patrick Wieschollek <mail@patwie.com>

#ifndef FIELD_T_H
#define FIELD_T_H

#include <iostream>
#include <memory>

#include "token_t.h"

class group_t;

class field_t {
  public:

    field_t();

    const token_t token() const;

    void token(const token_t tok);
    void pos(int x, int y);

    const int x() const;
    const int y() const;

    friend std::ostream& operator<< (std::ostream& stream, const field_t& stone);

    group_t* group;
    int played_at;

  private:
    int x_;
    int y_;
    token_t token_;
};

#endif