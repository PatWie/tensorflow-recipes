import re
from gtp_bridge import GTPbridge


class BaseEngine(object):
    """A wrapper for the GTP class.
    """
    def tuple2string(self, pos):
        """Convert tuple of position (x, y) into GTP moves

        Remarks:
            (0,1) is A1

        Args:
            pos ((int, int)): Description

        Returns:
            str: position description
        """
        charset = "ABCDEFGHJKLMNOPQRST"
        x, y = pos
        return '%s%s' % (charset[x - 1], y)

    def string2tuple(self, ans):
        """Parse a GTP position string into a tuple

        Args:
            ans (str): move encoded as string

        Returns:
            (int, int): decoded move
        """
        ans = self.clean(ans)

        x = ans[0]
        y = ans[1:]
        charset = "ABCDEFGHJKLMNOPQRST"
        x = charset.find(x.upper()) + 1

        return x, int(y)

    def clean(self, s):
        """Remove uninteresting characters from GTP message.

        Args:
            s (str): message

        Returns:
            str: cleaned message
        """
        s = re.sub("[\t\n =]", "", s)
        return s

    def __init__(self, name, pipe_args):
        """Initialize new GTP app.

        Args:
            name (str): Name of GO entity
            pipe_args (list(str)): command to start GTP app
        """
        self.bridge = GTPbridge(name, pipe_args)
        pass

    def name(self):
        return self.bridge.send("name\n")

    def show_board(self):
        return self.bridge.send('showboard\n')

    def boardsize(self, boardsize=19):
        return self.bridge.send("boardsize {}\n".format(boardsize))

    def clear_board(self):
        return self.bridge.send("clear_board\n")

    def parse_move(self, ans):
        ans = self.clean(ans)
        hint = None
        if ans.lower() == 'pass':
            hint = 'pass'
            return 0, 0, 'pass'
        if ans.lower() == 'resign':
            hint = 'resign'
            return 0, 0, 'resign'

        x, y = self.string2tuple(ans)
        return x, y, hint

    def generate_move(self, color):
        """Generate and Play move for given color.

        Args:
            color (char): either 'W' or 'B'

        Returns:
            (int, int, str): position (x, y) and flag (pass, resign, None)
        """
        assert color in ['W', 'B']
        ans = self.bridge.send("genmove {}\n".format(color))
        return self.parse_move(ans)

    def play(self, color, pos):
        assert color in ['W', 'B']
        self.bridge.send("play {} {}\n".format(color, self.tuple2string(pos)))

    def score(self):
        return self.bridge.send("final_score\n")

    def komi(self, komi):
        return self.bridge.send("komi %s\n" % komi)
