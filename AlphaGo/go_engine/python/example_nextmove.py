#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>


import goplanes
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--steps', help='number of moves to play', type=np.int32, default=10)
args = parser.parse_args()


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/game.sgfbin')
assert os.path.isfile(path)

# Version from file
# --------------------------------------------------------
max_moves = os.path.getsize(path) / 2
planes = np.zeros((47 * 19 * 19), dtype=np.int32)
steps = min(int(args.steps), max_moves - 1)
next_move = goplanes.planes_from_file(path, planes, steps)
planes = planes.reshape((47, 19, 19))

board = np.zeros((19, 19), dtype=str)
board[planes[0, :, :] == 1] = 'o'
board[planes[1, :, :] == 1] = 'x'
board[planes[2, :, :] == 1] = '.'

for i in range(19):
    print " ".join(board[i, :])
print "max", max_moves
print "next", next_move

# Version from bytes
# --------------------------------------------------------
raw = np.fromfile(path, dtype=np.int8)
max_moves = len(raw) / 2

planes = np.zeros((47 * 19 * 19), dtype=np.int32)
steps = min(int(args.steps), max_moves - 1)
next_move = goplanes.planes_from_bytes(raw.tobytes(), planes, steps)
planes = planes.reshape((47, 19, 19))

bboard = np.zeros((19, 19), dtype=str)
bboard[planes[0, :, :] == 1] = 'o'
bboard[planes[1, :, :] == 1] = 'x'
bboard[planes[2, :, :] == 1] = '.'

for i in range(19):
    print " ".join(bboard[i, :])
print "max", max_moves
print "next", next_move
