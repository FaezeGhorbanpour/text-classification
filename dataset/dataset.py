import collections
import math


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
        r = round(math.log10(len(vocabs)))
        return 10 ** r


    def get_train(self):
        return self.train_x, self.train_y

    def get_test(self):
        return self.test_x, self.test_y

    def get_validation(self):
        return self.validation_x, self.validation_y
