from gtp_engine_base import BaseEngine
from gtp_engine_gnugo import GnuGoEngine

import sys
sys.path.insert(0, "../python")  # noqa

import goplanes
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class TfGoEngine(BaseEngine):

    def __init__(self, name, export_dir):
        """Init Tensorflow Go Engine

        Args:
            name (str): name of entity
            export_dir (str): path to exported tensorflow model
        """
        assert os.path.isdir(export_dir)
        # we will use gnugo to guide our network
        self.assistant = GnuGoEngine('assistant', verbose=False)
        # create session and load network
        self.sess = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True))
        tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], export_dir)

        # get input node, output node
        self.features = self.sess.graph.get_tensor_by_name('board_plhdr:0')
        self.prob = self.sess.graph.get_tensor_by_name('probabilities:0')

    def get_probabilities(self, planes):
        # run NN prediction and return softmax(logits)
        return self.sess.run(self.prob, {self.features: planes})[0][0]

    def debug(self, prob, planes):
        charset = "ABCDEFGHJKLMNOPQRST"
        legend_a = ['    %s   ' % c for c in charset]
        print('   ' + "".join(legend_a))

        debug_prob = ""
        for ix in range(19):
            debug_prob += '%02i ' % (19 - ix)
            for iy in range(19):
                tok = '.'
                if planes[0, 0, ix, iy] == 1:
                    tok = 'y'
                if planes[0, 1, ix, iy] == 1:
                    tok = 'n'
                # debug_prob = "%s %04.2f(%s)" % (debug_prob, prob[ix][iy], tok)
                debug_prob += '  .%02i(%s)' % (int(prob[ix][iy] * 100), tok)
            debug_prob += '  %02i ' % (19 - ix)
            debug_prob += "\n"
        print(debug_prob + '   ' + "".join(legend_a))

    def generate_move(self, color):

        def flat2plane(c):
            x = c // 19
            y = c % 19
            return x, y

        assert color in ['W', 'B']

        # ask GnuGo if we should pass or resign
        # TODO: replace by value network
        _, _, hint = self.assistant.propose_move(color)

        if hint == 'pass':
            return 0, 0, 'pass'

        if hint == 'resign':
            return 0, 0, 'resign'

        # ask GnuGo for a current board configuration
        # TODO: not sure, if we can keep the C++ structures from goplane in memory between python calls
        #       but GnuGo is fast enough to build the board
        white_board, black_board = self.assistant.get_board()

        # extract features for this board configuration
        planes = np.zeros((47, 19, 19), dtype=np.int32)
        goplanes.planes_from_position(white_board, black_board, planes, int(color == 'W'))
        planes = planes.reshape((1, 47, 19, 19))

        # predict a move
        prob = self.get_probabilities(planes)
        print('TfGO probabilities:\n')
        self.debug(prob, planes)

        prob = prob.reshape(19 * 19)

        # sort moves according probability (high to low)
        candidates = np.argsort(prob)[::-1]

        print "TfGO Top-10 predictions (legal moves):"

        found_legal_moves = 0
        for i in range(60):
            c = candidates[i]
            p = prob[c]
            x, y = flat2plane(c)
            # WHY ?????????????
            x, y = y + 1, 19 - x
            if self.assistant.is_legal(color, (x, y)):
                found_legal_moves += 1
                print "%03i:  %s \t(%f)" % (i + 1, self.tuple2string((x, y)), p)
                if found_legal_moves == 10:
                    break

        for c in candidates:
            p = prob[c]
            x, y = flat2plane(c)
            x, y = y + 1, 19 - x
            is_legal = 'legal' if self.assistant.is_legal(color, (x, y)) else 'illegal'
            if is_legal == 'legal':
                print "TfGo plays  %s \t(%f)" % (self.tuple2string((x, y)), p)
                self.assistant.play(color, (x, y))
                break

        # raw_input("Press Enter to continue...")
        return x, y, None

    def play(self, color, pos):
        self.assistant.play(color, pos)

    # override
    def name(self):
        return 'TfGo'

    def show_board(self):
        return self.assistant.show_board()

    def boardsize(self, boardsize=19):
        return self.assistant.boardsize(boardsize)

    def clear_board(self):
        return self.assistant.clear_board()

    def score(self):
        return self.assistant.score()

    def komi(self, komi):
        return self.assistant.komi(komi)
