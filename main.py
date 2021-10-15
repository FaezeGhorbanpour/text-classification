import argparse
import tensorflow as tf

from datasets.twitter.dataset_loader import TwitterLoader
from datasets.weibo.dataset_loader import WeiboLoader
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
from models.transformer_models.bert import Bert
from models.transformer_models.xlnet import Xlnet

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
        raise Exception('Invalid datasets name!')

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
    elif args.model == 'bert':
        model = Bert(embedding)
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

