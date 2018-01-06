from gtp_engine_gnugo import GnuGoEngine
import numpy as np
import goplanes

a = GnuGoEngine('gnugoA')

# print a.is_legal('W', (4, 4))
# a.play('W', (4, 4))
# print a.is_legal('W', (4, 4))


a.play('W', (4, 4))

board = a.show_board()


def get_board(board):
    import re           # noqa
    import numpy as np  # noqa
    regex = r"[0-9]{1,2}([ .+OX]*)[0-9]{1,2}"
    matches = re.finditer(regex, board)

    matches = [m.group(1).replace('+', '.').replace(' ', '') for m in matches]

    white = np.zeros((19, 19)).astype(np.int32)
    black = np.zeros((19, 19)).astype(np.int32)

    for x, line in enumerate(matches):
        for y, tok in enumerate(line):
            if tok == 'O':
                white[x][y] = 1
            if tok == 'X':
                black[x][y] = 1

    return white, black


w, b = get_board(board)
color = 'W'

planes = np.zeros((47 * 19 * 19), dtype=np.int32)
goplanes.planes_from_position(w, b, planes, int(color == 'W'))
planes = planes.reshape((47, 19, 19))
