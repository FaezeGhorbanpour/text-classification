from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import jieba
from nltk.tokenize import word_tokenize
from embeddings.embedding import Embedding


class Tfidf(Embedding):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.embedding_name = 'tfidf'
        if dataset.language == 'en':
            tokenizer = word_tokenize
        elif dataset.language == 'chi':
            tokenizer = jieba.lcut
        else:
            tokenizer = None
        self.vect = TfidfVectorizer(strip_accents='unicode', tokenizer=tokenizer, ngram_range=(1, 4), max_df=0.75, min_df=3,
                               sublinear_tf=True,)

        self.train_x = self.vect.fit_transform(dataset.train_x)
        self.test_x = self.vect.transform(dataset.test_x)
        self.validation_x = self.vect.transform(dataset.validation_x)

