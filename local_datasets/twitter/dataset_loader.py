from local_datasets.dataset import Dataset
import pandas as pd
import os


class TwitterLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'twitter'
        self.data_path = '../../../../../media/external_3TB/3TB/ghorbanpoor/twitter/'
        self.output_path = 'local_datasets/twitter/'
        self.language = 'en'
        self.labels_name = ['real', 'fake']

        train = pd.read_csv(os.path.join(self.data_path, "twitter_train_translated.csv"))
        self.train_x = train['text'].values
        self.train_y = train['label'].values
        test = pd.read_csv(os.path.join(self.data_path, "twitter_test_translated.csv"))
        self.test_x = test['text'].values
        self.test_y = test['label'].values
        validation = pd.read_csv(os.path.join(self.data_path, "twitter_test_translated.csv"))
        self.validation_x = validation['text'].values
        self.validation_y = validation['label'].values

        self.max_words = self.get_max_words()
        self.max_length = 32


if __name__ == '__main__':
    data = TwitterLoader()
    vocabs = data.get_vocabs()
    print(vocabs)