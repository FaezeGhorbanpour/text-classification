import collections

from local_datasets.dataset import Dataset
import pandas as pd
import os

import jieba.posseg as pseg




class WeiboLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'weibo'
        # self.data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/weibo/'
        self.data_path = '/home/faeze/PycharmProjects/fake_news_detection/data/weibo/'
        self.output_path = 'local_datasets/weibo/'
        self.language = 'chi'
        self.labels_name = ['real', 'fake']

        train = pd.read_csv(os.path.join(self.data_path, "weibo_train_text.csv"), header=None)
        train.columns = ['id', 'text', 'label']
        train = train.dropna()
        self.train_x = train['text'].values
        self.train_y = train['label'].values
        test = pd.read_csv(os.path.join(self.data_path, "weibo_test_text.csv"), header=None)
        test.columns = ['id', 'text', 'label']
        test = test.dropna()
        self.test_x = test['text'].values
        self.test_y = test['label'].values
        validation = pd.read_csv(os.path.join(self.data_path, "weibo_test_text.csv"), header=None)
        validation.columns = ['id', 'text', 'label']
        validation = validation.dropna()
        self.validation_x = validation['text'].values
        self.validation_y = validation['label'].values

        self.max_words = 30000
        self.max_length = 200

    def get_vocabs(self):
        def update_vocab_counter(row):
            for i in row:
                vocab_counter[i.word] += 1

        vocab_counter = collections.Counter()
        [update_vocab_counter(pseg.cut(i)) for i in self.train_x]
        vocabs = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
        return vocabs






if __name__ == '__main__':
    data = WeiboLoader()


