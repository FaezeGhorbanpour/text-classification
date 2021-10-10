import collections
import math

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

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


    def tokenizer(self):

        tokenizer_obj = Tokenizer()

        tokenizer_obj.fit_on_texts(self.train_x)

        train_x_tokens = tokenizer_obj.texts_to_sequences(self.train_x)
        test_x_tokens = tokenizer_obj.texts_to_sequences(self.test_x)
        validation_x_tokens = tokenizer_obj.texts_to_sequences(self.validation_x)

        self.train_x = pad_sequences(train_x_tokens, maxlen=self.max_length, padding='post')
        self.test_x = pad_sequences(test_x_tokens, maxlen=self.max_length, padding='post')
        self.validation_x = pad_sequences(validation_x_tokens, maxlen=self.max_length, padding='post')