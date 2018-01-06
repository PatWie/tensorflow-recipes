
import sys
import glob
sys.path.insert(0, "../go_engine/gtp/")  # noqa
sys.path.insert(0, "../go_engine/python/")  # noqa

import numpy as np
import re
from gtp_bridge import GTPbridge
import goplanes
import os

fn = "/home/patwie/godb/Database/2000/2000-00-00a.sgf"
fn = "/home/patwie/godb/Database/test.sgf"


def get_gnugo_board(fn, until=None):
    """Use GnuGO to compute final board of game.

    Args:
        fn (str): path to -sgf file
        until (str, optional): number of moves to play (exclude handicap stones)

    Returns:
        (np.aray, np.aray): board white, black
    """
    if until is not None:
        print('[gnugo] play %i moves' % until)
        until = ['-L', str(until)]
    else:
        until = []
    bridge = GTPbridge('name', ["gnugo",
                                "--infile", fn,
                                "--mode", "gtp"] + until, verbose=True)

    board = bridge.send('showboard\n')

    def get_board(board):
        """Return NumPy arrays of the current board configuration.

        Remarks:
            This parses thr GnuGO output (which might change over the versions).
            see: https://regex101.com/r/VvOM0Y/1

        Returns:
            (np.array, np.array): all placed white tokens, all placed black tokens
        """
        regex = r"[0-9]{1,2}([ .+OX]*)[0-9]{1,2}"
        lines = board.split('\n')
        lines = [l[:43] for l in lines]
        board = "\n".join(lines)
        matches = re.finditer(regex, board)

        matches = [m.group(1).replace('+', '.').replace(' ', '') for m in matches]

        white = np.zeros((19, 19)).astype(np.int32)
        black = np.zeros((19, 19)).astype(np.int32)

        for x, line in enumerate(matches):
            for y, tok in enumerate(line):
                if tok == 'X':
                    black[x][y] = 1
                if tok == 'O':
                    white[x][y] = 1

        return white, black

    ans = get_board(board)
    bridge.close()
    return ans


def print_board(white, black=None):
    """Produce GnuGO like output to verify board position.

    Args:
        white (np.array): array with 1's for white
        black (np.array): array with 1's for black

    Returns:
        str: gnugo like output (without legend)
    """

    if black is None:
        n = np.copy(white)
        white = n < 0
        black = n > 0

    s = ''
    charset = "ABCDEFGHJKLMNOPQRST"
    s += '   '
    for x in xrange(19):
        s = s + charset[x] + ' '
    s += '\n'
    for x in xrange(19):
        s += '%02i ' % x
        for y in xrange(19):
            if white[x][y] == 1:
                s += '0 '
            elif black[x][y] == 1:
                s += 'X '
            else:
                s += '. '
        s += ' %02i' % x
        s += '\n'
    charset = "ABCDEFGHJKLMNOPQRST"
    s += '   '
    for x in xrange(19):
        s = s + charset[x] + ' '
    s += '\n'
    return s


def get_own_board(fn, until=None):
    """Use goplanes library from this repository for board representation.

    Remarks:
        This uses the feature-plane code (input for NN) to create the board positions.
        We just use it for testing. So far all final board configurations from `GoGoD`
        matches the ones from GnuGO.

    Args:
        fn (str): path to *.sgfbin file
        until (int, optional): number of moves to play (with handicap stones)

    Returns:
        (np.array, np.array): all placed white tokens, all placed black tokens
    """
    max_moves = os.path.getsize(fn + 'bin') / 2
    print('[own] has %i moves' % max_moves)
    planes = np.zeros((47, 19, 19), dtype=np.int32)
    if until is not None:
        until = max(0, until + 2)
    else:
        until = -1
    print('[own] play %i moves' % until)
    goplanes.planes_from_file(fn + 'bin', planes, until)

    from_black = (planes[-1][0][0] == 1)
    if from_black:
        actual_b = planes[0]
        actual_w = planes[1]
    else:
        actual_b = planes[1]
        actual_w = planes[0]

    return actual_w, actual_b


# for fn in glob.glob('/home/patwie/godb/Database/1998/1998-03-09d.sgf'):
# for fn in glob.glob('/home/patwie/godb/Database/1998/1998-01-09k.sgf'):
for fn in glob.glob('/home/patwie/godb/Database/*/*.sgf'):

    # filter not converted games (like incorrect and amateur)
    if not os.path.isfile(fn):
        continue
    if not os.path.isfile(fn + 'bin'):
        continue
    print fn
    moves = None  # means all
    # moves = 50
    # moves = os.path.getsize(fn + 'bin') / 2
    expected_w, expected_b = get_gnugo_board(fn, moves)
    actual_w, actual_b = get_own_board(fn, moves)

    # if True:
    if (expected_w - actual_w).sum() > 0 or (expected_b - actual_b).sum() > 0:
        # there is a difference between GnuGO and GoPlanes
        print fn
        print print_board(expected_w, expected_b)
        print print_board(actual_w, actual_b)
        print (expected_w - actual_w).sum()
        print (expected_b - actual_b).sum()

        print print_board(expected_w - actual_w)
        print print_board(expected_b - actual_b)
        raw_input("prompt")
