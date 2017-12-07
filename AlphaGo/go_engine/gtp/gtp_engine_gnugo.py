from gtp_engine_base import BaseEngine
from gtp_bridge import GTPbridge

import numpy as np
import re


class GnuGoEngine(BaseEngine):
    # see http://www.delorie.com/gnu/docs/gnugo/gnugo_207.html
    def __init__(self, name, verbose=True):
        self.bridge = GTPbridge(name, ["gnugo", "--mode", "gtp"], verbose=verbose)

    def is_legal(self, color, pos):
        """Check whether a token can be placed at a field.

        Args:
            color (str): either 'B' or 'W'
            pos ((int, int)): field to play

        Returns:
            bool: valid move?
        """
        assert color in ['W', 'B']
        ans = self.bridge.send("is_legal {} {}\n".format(color, self.tuple2string(pos)))
        ans = int(self.clean(ans))
        return (ans == 1)

    def propose_move(self, color):
        """Propose a move without playing it.

        Args:
            color (str): Description

        Returns:
            TYPE: Description
        """
        assert color in ['W', 'B']
        move = self.bridge.send('gg_genmove {}\n'.format(color))
        return self.parse_move(move)

    def get_board(self):
        """Return NumPy arrays of the current board configuration.

        Returns:
            (np.array, np.array): all placed white tokens, all placed black tokens
        """
        board = self.show_board()
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
