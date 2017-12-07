import os
import logging
import sys
import glob
import argparse
import numpy as np
logging.basicConfig(stream=sys.stdout, level=logging.WARN)

logger = logging.getLogger('sgfreader')


class SGFMeta(object):
    """Some information about the match."""
    def __init__(self, tokens):
        super(SGFMeta, self).__init__()
        self.tokens = tokens
        self.info = dict()

        # keys for meta data
        keys = ['SZ', 'PW', 'WR', 'PB', 'BR', 'FF',
                'EV', 'RO', 'DT', 'PC', 'KM', 'RE',
                'GC', 'US', 'RU', 'WT', 'BT', 'TM']

        # set default
        for k in keys:
            self.info[k] = ''

        # read only tokens with key from 'keys'
        for k, v in tokens:
            if k in keys:
                self.info[k] = str(v)

    def summarize(self):
        d = self.info
        print '%s (%s) vs. %s (%s)' % (d['PW'], d['WR'], d['PB'], d['BR'])
        print 'on %s board size %s' % (d['EV'], d['SZ'])
        print 'comment: %s' % (d['GC'])
        print 'has timelimit: %s' % (d['TM'] is not '')

    @property
    def correct(self):
        """An attempt to test correctness of match.

        Remarks:
            The game commentary (GC) or result (RE) contains
            valueable information for this.

        Returns:
            TYPE: Description
        """
        if self.info['SZ'] != '19':
            return False
        if 'illegal' in self.info['GC'].lower():
            return False
        if 'corrupt' in self.info['GC'].lower():
            return False
        if 'time' in self.info['RE'].lower():
            return False
        if 'resign' in self.info['RE'].lower():
            return False
        return True


class SGFReader(object):
    """Read plaintext SGF file."""
    def __init__(self, fn):
        assert os.path.isfile(fn)
        self.fn = fn
        super(SGFReader, self).__init__()
        # read raw data
        with open(fn, 'r') as f:
            self.content = f.read()
        self.meta = None
        # store all moves here
        self.moves = []
        self.move_len = 0
        self.result = ''
        # start parsing
        self.parse()

    @property
    def amateur(self):
        return 'amateur' in self.content.lower()

    def parse(self):
        # these tokens separete the information
        separators = set(['(', ')', '\n', '\r', '\t', ';', ']'])

        tokens = []
        k, v, is_key = '', '', True
        k_prev = None
        for c in self.content:
            if c in separators:
                # new piece of information is complete?
                if v is not '':
                    # treat handicap with single-key multi value 'AB[dp][pd][pp]'
                    if k is '':
                        k = k_prev
                    tokens.append((k.strip(), v.strip()))
                    k_prev = k
                # reset state
                k, v, is_key = '', '', True
                continue
            # trigger value reader
            if c == '[':
                is_key = False
                continue
            # append character to state
            if is_key:
                k += str(c)
            else:
                v += str(c)

        """
        Move Properties             B, KO, MN, W
        Setup Properties            AB, AE, AW, PL
        Node Annotation Properties  C, DM, GB, GW, HO, N, UC, V
        Move Annotation Properties  BM, DO, IT, TE
        Markup Properties           AR, CR, DD, LB, LN, MA, SL, SQ, TR
        Root Properties             AP, CA, FF, GM, ST, SZ
        Game Info Properties        AN, BR, BT, CP, DT, EV, GN, GC, ON, OT, PB, PC, PW, RE, RO, RU, SO, TM, US, WR, WT
        Timing Properties           BL, OB, OW, WL
        Miscellaneous Properties    FG, PM, VW
        """

        valid_keys = ['B', 'KO', 'MN', 'W',
                      'AB', 'AE', 'AW', 'PL',
                      'C', 'DM', 'GB', 'GW', 'HO', 'N', 'UC', 'V',
                      'BM', 'DO', 'IT', 'TE',
                      'AR', 'CR', 'DD', 'LB', 'LN', 'MA', 'SL', 'SQ', 'TR',
                      'AP', 'CA', 'FF', 'GM', 'ST', 'SZ',
                      'AN', 'BR', 'BT', 'CP', 'DT', 'EV', 'GN', 'GC', 'ON', 'OT', 'PB', 'PC', 'PW', 'RE', 'RO', 'RU', 'SO', 'TM', 'US', 'WR', 'WT',  # noqa
                      'BL', 'OB', 'OW', 'WL',
                      'FG', 'PM', 'VW',
                      'KM', 'OH', 'HA', 'MULTIGOGM']
        # I do not the keys from the last row and didn't find any information on the internet.
        # But they are in the data.

        # pass over all tokens from SGF
        for k, v in tokens:
            if k not in valid_keys:
                logger.warn('{} is not a valid key in {}'.format(k, self.fn))
                logger.warn('{}'.format(v))
                sys.exit(0)

            if k == 'KO':
                # this should not happen, as we filter out these games beforehand
                logger.error('illegal move in %s' % self.fn)

            # play move (W, B) or set stone (AW, AB)
            if k in ['AB', 'AW', 'B', 'W']:
                self.moves.append((k, v))
            # count moves for statistics
            if k in ['B', 'W']:
                self.move_len += 1

            if k == 'RE':
                # parse result of game
                if v.strip() == '0':
                    self.result = 'drawn'
                if 'drawn' in v.strip().lower():
                    self.result = 'drawn'
                if 'jigo' in v.strip().lower():
                    self.result = 'drawn'
                if 't+' in v.strip().lower():
                    logger.error('time resign')
                ok = True
                # result should start with W, B like (W+2.5)
                if not v.strip().lower().startswith('b'):
                    if not v.strip().lower().startswith('w'):
                        # logger.error(v)
                        ok = False
                if ok:
                    if v.strip().lower().startswith('b'):
                        self.result = 'black'
                    else:
                        self.result = 'white'

        self.meta = SGFMeta(tokens)


def encode(color, x, y, move=False, pass_move=False):
    """Encode all moves into 16bits (easier to read in cpp and binary!!)

    ---pmcyyyyyxxxxx (4bits left for additional information)

    x encoded column
    y encoded row
    c color [w=1, b=0]
    m isemove [move=1, add=0]
    p ispass  [pass=1, nopass=0]

    TODO:
        handle "pass move"

    Args:
        color (str): Description
        x (int): Description
        y (int): Description
        move (bool, optional): Description

    Returns:
        (int, int): Description
    """
    assert color in ['W', 'B', 'AW', 'AB']
    # assert x < 19
    # assert y < 19

    value = 32 * y + x
    if color in ['W', 'AW']:
        value += 1024
    if move:
        value += 2048
    if pass_move:
        value += 4096

    byte1 = value % (2**8)
    byte2 = (value - byte1) / (2**8)
    # print('value %i' % value)
    # print('encode %i/%i is_move %i as [%i %i]' % (x, y, move, byte1, byte2))

    return [byte2, byte1]


def debug(pattern):
    corrupted_files = []
    good_files = []
    amateur_files = []
    total_moves = 0

    files = glob.glob(pattern)
    for fn in files:
        if '1700-99' in fn:
            corrupted_files.append(fn)
            logger.warn('{} is too old'.format(fn))
            continue
        if '0196-1699' in fn:
            corrupted_files.append(fn)
            logger.warn('{} is too old'.format(fn))
            continue

        s = SGFReader(fn)
        if s.meta.correct:
            if s.amateur:
                amateur_files.append(fn)
            else:
                # the interesting ones
                total_moves += s.move_len
                good_files.append(fn)

        else:
            logger.warn('{} is not correct'.format(fn))
            logger.info('--> %s' % s.meta.info['GC'])
            corrupted_files.append(fn)

    def out_of(caption, needle, haystack):
        print '%i out of %i are %s ~%f%%' % (len(needle), len(files), caption, len(needle) / float(len(haystack)) * 100)

    out_of('corrupt', corrupted_files, files)
    out_of('amateur', amateur_files, files)

    print 'total moves %i' % total_moves
    print 'avg moves %i' % (total_moves / float(len(good_files)))


def convert(pattern):
    files = glob.glob(pattern)
    for fn in files:
        logger.info('convert {}'.format(fn))
        if '1700-99' in fn:
            logger.info('{} is too old, we skip'.format(fn))
            continue
        if '0196-1699' in fn:
            logger.info('{} is too old, we skip'.format(fn))
            continue

        s = SGFReader(fn)
        if s.meta.correct:
            if not s.amateur:
                # the interesting ones
                bin_content = []
                for k, v in s.moves:

                    is_set = False
                    if k in ['AB', 'AW']:
                        is_set = True

                    # print k
                    if k in ['B', 'AB']:
                        color = 'B'
                    else:
                        color = 'W'

                    if len(v.strip()) == 0 or v.strip().lower() == 'tt':
                        # pass
                        bin_content += encode(color, 0, 0, move=False, pass_move=True)
                    else:
                        if len(v) != 2:
                            # print k, v, fn
                            logger.error("cannot parse {} in {}".format(v, fn))
                        else:
                            # ok move
                            x, y = v.lower()
                            # print "move (%s) at %s %s" % (k, x, y)
                            x, y = ord(x) - ord('a'), ord(y) - ord('a')
                            bin_content += encode(color, x, y, move=not is_set, pass_move=False)
                yield [bin_content, fn]
            else:
                logger.info('game is game of only amateurs')
        else:
            logger.warn('{} is not correct'.format(fn))
            logger.warn('--> %s' % s.meta.info['GC'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='', default='debug', type=str)
    parser.add_argument('--pattern', help='', default='/home/patwie/godb/Database/*/*.sgf')
    args = parser.parse_args()

    # just for testing and some statistics
    if args.action == 'debug':
        debug(args.pattern)
        sys.exit(0)

    # convert sgf to binary
    if args.action == 'convert':
        for bin_content, fn in convert(args.pattern):
            bin_content = np.array(bin_content).astype(np.uint8)
            with open('%sbin' % fn, 'wb') as f:
                f.write(bin_content.tobytes())
        sys.exit(0)
