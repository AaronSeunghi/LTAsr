from __future__ import absolute_import, division, print_function

import codecs
import numpy as np
import tensorflow as tf
import re

from six.moves import range
from functools import reduce

# The following words and indexes are fixed
# <pad> : 0 
# <sos> : 1
# <eos> : 2

class Alphabet(object):
    def __init__(self, config_file):
        self._label_to_str = []
        self._str_to_label = {}
        self._size = 0
        with codecs.open(config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str += line[:-1] # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1

    def string_from_label(self, label):
        return self._label_to_str[label]

    def label_from_string(self, string):
        return self._str_to_label[string]

    def size(self):
        return self._size

def text_to_char_array(original, alphabet):
    r"""
    Given a Python string ``original``, remove unsupported characters, map characters
    to integers and return a numpy array representing the processed string.
    """
    return np.asarray([alphabet.label_from_string(c) for c in original])

def get_start_sym_label(alphabet):
    return np.asarray([alphabet.label_from_string('<sos>')])

def get_end_sym_label(alphabet):
    return np.asarray([alphabet.label_from_string('<eos>')])


