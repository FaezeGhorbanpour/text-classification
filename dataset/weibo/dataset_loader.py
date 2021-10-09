from dataset.dataset import Dataset
import pandas as pd
import os

class WeiboLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.data_name = 'twitter'
        self.data_path = '../../fake_news_detection/data/weibo/'
        self.language = 'en'

        train = pd.read_csv(os.path.join(self.data_path, "weibo_train_text.csv"), header=None)
        train.columns = ['id', 'text', 'label']
        self.train_x = train['text'].values
        self.train_y = train['label'].values
        test = pd.read_csv(os.path.join(self.data_path, "weibo_test_text.csv"), header=None)
        test.columns = ['id', 'text', 'label']
        self.test_x = test['text'].values
        self.test_y = test['label'].values
        validation = pd.read_csv(os.path.join(self.data_path, "weibo_test_text.csv"), header=None)
        validation.columns = ['id', 'text', 'label']
        self.validation_x = validation['text'].values
        self.validation_y = validation['label'].values




if __name__ == '__main__':
    data = WeiboLoader()


