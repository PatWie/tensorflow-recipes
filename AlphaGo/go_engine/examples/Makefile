all: build

build: clean main

examples: basic

basic: basic.cpp
	clang++ -std=c++11 basic.cpp ../src/field_t.cpp ../src/group_t.cpp ../src/board_t.cpp -I ../src -o basic

clean:
	rm -f *.o *.so *.pyc *.npy