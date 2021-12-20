from local_datasets.dataset import Dataset
import pandas as pd
import os
import torch


class SentimentLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'sentiment'
        self.data_path = '/home/faeze/sentiment-classification/local_datasets/sentiment/data/'
        self.output_path = '/home/faeze/sentiment-classification/local_datasets/sentiment/'
        self.language = 'fa'
        self.labels_name = ['positive', 'neutral', 'negative']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        train = train.dropna()
        train.columns = ['text', 'label']
        test = pd.read_csv(os.path.join(self.data_path, "test.csv"))
        test = test.dropna()
        test.columns = ['text', 'label']
        try:
            validation = pd.read_csv(os.path.join(self.data_path, "validation.csv"))
            validation = validation.dropna()
            validation.columns = ['text', 'label']
        except:
            offset = int(0.9*train.shape[0])
            validation = train[offset:]
            train = train[:offset]


        self.train_x = train['text'].values
        self.train_y = train['label'].values
        self.test_x = test['text'].values
        self.test_y = test['label'].values
        self.validation_x = validation['text'].values
        self.validation_y = validation['label'].values

        self.max_words = 400000
        self.max_length = 100


if __name__ == '__main__':
    data = SentimentLoader()
    vocabs = data.get_vocabs()
    print(vocabs)


