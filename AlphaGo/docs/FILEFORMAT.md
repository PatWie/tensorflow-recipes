# Fileformat

In contrast to chess (8x8) we cannot store a board configuration into `uin64` datatypes (one for each figure). As computing the GO-features for the NN requires to compute liberties of groups all the time, we store the moves into a binary format and replay them. The best I came up with is 2bytes for each single move:

```
---pmcyyyyyxxxxx

p: is action passed ? [1:yes, 0:no]
m: is action a move ? [1:yes, 0:no] (sgf supports 'set' as well)
c: is action from white ? [1:yes, 0:no] (0 means black ;-) )
y: encoded row (1-19)
x: encoded column (a-s)
-: free bits (maybe we can store something useful here)
```

This gives two nice properties:
- Reading a match and its moves in C++ is absolute easy.
- Computing the length of the game is simply `sizeof(file) / 2` (we just ignore the handicap currently)