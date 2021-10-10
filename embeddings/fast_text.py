import numpy as np
from embeddings.embedding import Embedding
from models.fast_text import Fasttext


class FasttextEmbedding(Embedding):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.embedding_name = 'fasttext'

        self.vect = Fasttext(self)
        model = self.vect.train()
        self.train_x = np.array([model.get_sentence_vector(i) for i in dataset.train_x])
        self.test_x = np.array([model.get_sentence_vector(i) for i in dataset.test_x])
        self.validation_x = np.array([model.get_sentence_vector(i) for i in dataset.validation_x])




