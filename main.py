import argparse
import tensorflow as tf

from dataset.twitter.dataset_loader import TwitterLoader
from dataset.weibo.dataset_loader import WeiboLoader
from embeddings.tfidf import Tfidf
from models.catboost import Catboost
from models.fast_text import Fasttext
from models.light_gbm import LightGBM
from models.logistic_regression import Logistic
from models.lstm import Lstm
from models.multi_conv import MultiConv
from models.svm import Svm
from models.xgboost import Xgboost

if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--embed', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--use_optuna', type=int, required=False)
    parser.add_argument('--extra', type=str, required=False)

    args = parser.parse_args()

    dataset = None
    if args.data == 'twitter':
        dataset = TwitterLoader()
    elif args.data == 'weibo':
        dataset = WeiboLoader()
    else:
        print(dataset)
        raise Exception('Invalid dataset name!')

    embedding = None
    if args.embed == 'tfidf':
        embedding = Tfidf(dataset)
    else:
        print(embedding)
        raise Exception('Invalid embedding name!')

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
    else:
        print(model)
        raise Exception('Invalid model name!')

    if args.use_optuna:
        model.optuna_main(args.use_optuna)
        model.main()
    else:
        model.main()

