#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
The simple implementation of Sentence2Vec using Word2Vec.
"""

import sys
import os
from textual.word2vec import Word2Vec, Sent2Vec, LineSentence
from common import *

def getTextualFeature(text_reading_path, size, output_file):
    # Train and save the Word2Vec model for the text file.
    # Please note that, you can change the dimension of the resulting feature vector by modifying the value of 'size'.
    raw_lines = LineSentence(text_reading_path)
    ids = [l[0] for l in raw_lines]
    line_sentence = [l[1:] for l in raw_lines]

    output_file = output_file + str(size)
    model = Word2Vec(line_sentence, size=size, window=5, sg=0, min_count=5, workers=8)
    model.save(output_file + '.model')

    # Train and save the Sentence2Vec model for the sentence file.
    vec = Sent2Vec(line_sentence, model=model)
    vec.save_sent2vec_format(output_file + '.vec', ids)

    program = os.path.basename(sys.argv[0])


if __name__ == '__main__':
    sizes = [100, 200, 300, 400, 500]
    for size in sizes:
        getTextualFeature('data/vine-desc-training.txt', size, textual_train_file_name)
        getTextualFeature('data/vine-desc-validation.txt', size, textual_test_file_name)