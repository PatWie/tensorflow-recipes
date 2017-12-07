class Match(object):
    """Simulate a match between two GTP programms"""
    def __init__(self, player_white, player_black, komi=5.5):
        super(Match, self).__init__()
        self.player_white = player_white
        self.player_black = player_black

        self.player_white.komi(komi)
        self.player_black.komi(komi)

        self.passed = False
        self.finished = False

    def ply(self, p, verbose=True):
        """Make a move (ply).

        Args:
            p (str): player ('W' or 'B')

        Returns:
            TYPE: Description
        """
        if self.finished:
            return

        # generate and play move
        if p.upper() == 'W':
            x, y, h = self.player_white.generate_move('W')
        else:
            x, y, h = self.player_black.generate_move('B')

        # ok game ends here
        if h == 'resign':
            self.finished = True
            return

        # game only ends if both players pass
        if h == 'pass':
            if self.passed:
                # both player passed
                self.finished = True
            else:
                self.passed = True
            return
        else:
            self.passed = False

        # synchronize boards accross GTP applications
        # sending move to other GTP app
        if p.upper() == 'W':
            self.player_black.play('W', (x, y))
        else:
            self.player_white.play('B', (x, y))

        if verbose:
            # just to flood the terminal
            self.player_white.show_board()
            self.player_black.show_board()
