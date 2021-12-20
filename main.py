import argparse

from embeddings.embedding import Embedding
from embeddings.tokenizer import Tokenizing
from local_datasets.sentiment.dataset_loader import SentimentLoader
from local_datasets.twitter.dataset_loader import TwitterLoader
from local_datasets.weibo.dataset_loader import WeiboLoader
from embeddings.tfidf import Tfidf
from models.machine_learning_models.catboost import Catboost
from models.deep_learning_models.fast_text import Fasttext
from models.machine_learning_models.light_gbm import LightGBM
from models.machine_learning_models.logistic_regression import Logistic
from models.deep_learning_models.lstm import Lstm
from models.deep_learning_models.multi_conv import MultiConv
from models.machine_learning_models.svm import Svm
from models.machine_learning_models.xgboost import Xgboost
from models.transformer_models.albert import Albert
from models.transformer_models.mbert import mBertClassifier
from models.transformer_models.parsbert import ParsBertClassifier
from models.transformer_models.xlnet import Xlnet

if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--embed', type=str, required=False)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--use_optuna', type=int, required=False)
    parser.add_argument('--extra', type=str, required=False)

    args = parser.parse_args()

    dataset = None
    if args.data == 'twitter':
        dataset = TwitterLoader()
    elif args.data == 'weibo':
        dataset = WeiboLoader()
    elif args.data == 'sentiment':
        dataset = SentimentLoader()
    else:
        print(dataset)
        raise Exception('Invalid local_datasets name!')

    embedding = None
    if args.embed == 'tfidf':
        embedding = Tfidf(dataset)
    elif args.embed == 'tokenizer':
        embedding = Tokenizing(dataset)
    else:
        embedding = Embedding(dataset)
        # raise Exception('Invalid embedding name!')

    model = None
    if args.model == 'logistic':
        model = Logistic(embedding)
    elif args.model == 'svm':
        model = Svm(embedding)
    elif args.model == 'fasttext':
        model = Fasttext(embedding)
    elif args.model == 'xgboost':
        model = Xgboost(embedding)
    elif args.model == 'catboost':
        model = Catboost(embedding)
    elif args.model == 'light_gbm':
        model = LightGBM(embedding)
    elif args.model == 'lstm':
        model = Lstm(embedding)
    elif args.model == 'multi_conv':
        model = MultiConv(embedding)
    elif args.model == 'parsbert':
        model = ParsBertClassifier(embedding)
    elif args.model == 'mbert':
        model = mBertClassifier(embedding)
    elif args.model == 'albert':
        model = Albert(embedding)
    elif args.model == 'xlnet':
        model = Xlnet(embedding)
    else:
        print(model)
        raise Exception('Invalid model name!')

    if args.use_optuna:
        model.optuna_main(args.use_optuna)
        model.main()
    else:
        model.main()

