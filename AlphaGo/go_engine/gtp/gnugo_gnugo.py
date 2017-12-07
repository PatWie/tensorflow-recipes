from gtp_engine_gnugo import GnuGoEngine
from gtp_match import Match


a = GnuGoEngine('gnugoA')
b = GnuGoEngine('gnugoB')

p = Match(a, b)

while p.finished is not True:
    p.ply('W')
    p.ply('B')

    a.show_board()

a.score()
b.score()
