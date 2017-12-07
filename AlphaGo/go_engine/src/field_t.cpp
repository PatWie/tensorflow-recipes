// Author: Patrick Wieschollek <mail@patwie.com>

#include "field_t.h"

field_t::field_t() : token_(empty), group(nullptr), x_(-1), y_(-1), played_at(0) {}

const token_t field_t::token() const {
    return token_;
}

void field_t::token(const token_t tok) {
    token_ = tok;
}

void field_t::pos(int x, int y) {
    x_ = x;
    y_ = y;
}

const int field_t::x() const { return x_;}
const int field_t::y() const { return y_;}

std::ostream& operator<< (std::ostream& stream, const field_t& stone) {
    if (stone.token() == empty){
        if(   ((stone.x() == 3) && (stone.y() == 3))   ||
              ((stone.x() == 3) && (stone.y() == 9))   ||
              ((stone.x() == 3) && (stone.y() == 15))  ||
              ((stone.x() == 9) && (stone.y() == 3))   ||
              ((stone.x() == 9) && (stone.y() == 9))   ||
              ((stone.x() == 9) && (stone.y() == 15))  ||
              ((stone.x() == 15) && (stone.y() == 3))  ||
              ((stone.x() == 15) && (stone.y() == 9))  ||
              ((stone.x() == 15) && (stone.y() == 15)) 
          ){
            stream  << "+";
        }
        else{
            stream  << ".";
        }
    }
    if (stone.token() == white)
        stream  << "o";
    if (stone.token() == black)
        stream  << "x";
    return stream;
}