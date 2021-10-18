from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from embeddings.embedding import Embedding


class Tokenizing(Embedding):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.embedding_name = 'tokenizer'

        self.train_x, self.test_x, self.validation_x = self.tokenizer()

    def tokenizer(self):
        tokenizer_obj = Tokenizer(self.dataset.max_words)

        tokenizer_obj.fit_on_texts(self.dataset.train_x)
        train_x_tokens = tokenizer_obj.texts_to_sequences(self.dataset.train_x)
        test_x_tokens = tokenizer_obj.texts_to_sequences(self.dataset.test_x)
        validation_x_tokens = tokenizer_obj.texts_to_sequences(self.dataset.validation_x)

        tokenizerd_train_x = pad_sequences(train_x_tokens, maxlen=self.dataset.max_length, padding='post')
        tokenizerd_test_x = pad_sequences(test_x_tokens, maxlen=self.dataset.max_length, padding='post')
        tokenizerd_validation_x = pad_sequences(validation_x_tokens, maxlen=self.dataset.max_length,
                                                padding='post')
        return tokenizerd_train_x, tokenizerd_test_x, tokenizerd_validation_x
