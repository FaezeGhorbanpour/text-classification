import collections
import math
import pandas as pd
import numpy as np
import torch


class Dataset:
    def __init__(self):
        self.data_name = ''
        self.data_path = ''
        self.output_path = ''
        self.labels_name = []
        self.train_x = None
        self.test_x = None
        self.validation_x = None
        self.train_y = None
        self.test_y = None
        self.validation_y = None
        self.max_length = 50
        self.max_words = 3000

    def normalizer(self, text):
        return text

    def translator(self, text):
        return text

    def get_labels_count(self):
        return len(self.labels_name)

    def get_vocabs(self):
        def update_vocab_counter(row):
            for word in row:
                vocab_counter[word] += 1

        vocab_counter = collections.Counter()
        [update_vocab_counter(i.split()) for i in self.train_x]
        vocabs = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
        return vocabs

    def get_max_words(self):
        vocabs = self.get_vocabs()
        vocabs_number = len(vocabs)
        r = round(math.log10(vocabs_number))
        return vocabs_number//(10 ** (r-1)) * 10 ** (r-1)

    def get_max_length(self):
        train = pd.DataFrame({'text': self.train_x})
        train['len'] = train.text.apply(lambda x: len(x.split()))
        print('max length: ', max(train['len']))
        print('min length: ', min(train['len']))
        print('mean length: ', np.mean(train['len']))
        print('average length: ', np.average(train['len']))




    def get_train(self):
        return self.train_x, self.train_y

    def get_test(self):
        return self.test_x, self.test_y

    def get_validation(self):
        return self.validation_x, self.validation_y
