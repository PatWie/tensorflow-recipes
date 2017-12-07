#include <iostream>

#include "board_t.h"

int main(int argc, char const *argv[]) {
    // test group capturing

    board_t b;

    b.play(0, 1, black);
    b.play(0, 2, black);
    b.play(1, 0, black);
    b.play(1, 3, black);
    b.play(1, 1, white);
    b.play(1, 2, white);
    b.play(2, 1, black);

    std::cout << b << std::endl;
    b.turn = black;

    board_t *copy_b = b.clone();
    std::cout << (*copy_b) << std::endl;
    
    std::cout << std::endl << "-----------------" << std::endl << std::endl;

    b.play(2, 2, black);
    std::cout << b << std::endl;

    copy_b->play(2, 2, black);
    std::cout << (*copy_b) << std::endl;


    return 0;
}