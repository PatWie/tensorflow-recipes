from gtp_engine_gnugo import GnuGoEngine
from gtp_engine_tfgo import TfGoEngine
from gtp_match import Match


a = GnuGoEngine('gnugoA')
b = TfGoEngine('tfgoB', '../../export')

p = Match(a, b)

while p.finished is not True:
    p.ply('B')
    p.ply('W')


a.score()
b.score()
