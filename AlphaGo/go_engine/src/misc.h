// Author: Patrick Wieschollek <mail@patwie.com>

#ifndef ENGINE_MISC_H
#define ENGINE_MISC_H

const int N = 19;

#define map2line(x,y) (((y) * 19 + (x)))
#define map3line(n,x,y) (( (n*19*19) +  (y) * 19 + (x)))
#define valid_pos(x) (((x>=0) && (x < N)))

#endif