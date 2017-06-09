from __future__ import absolute_import, division, print_function, unicode_literals

import io

import numpy as np


class Vocabulary:
    def __init__(self):
        self.words = []
        self.word_to_id_dict = {}

        # Add some special tokens
        self.add_words(['<PAD>', '<UNK>', '<GO>', '<EOS>'])

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_id_dict:
            self.words.append(word)
            new_id = len(self.words) - 1
            self.word_to_id_dict[word] = new_id

    def id_to_word(self, id_):
        return self.words[id_]

    def word_to_id(self, word):
        return self.word_to_id_dict.get(word, 1)  # 1 is the ID for <UNK>

    def add_from_file(self, path, max_words=np.inf):
        word_count = 0
        with io.open(path, encoding='utf-8') as file:
            for line in file:
                word, count = line.split()
                self.add_word(word)

                word_count += 1
                if word_count >= max_words:
                    break
