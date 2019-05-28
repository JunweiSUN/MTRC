import os
import json
import jieba
import shutil
import logging
import random
import time
from langconv import *
from zhon.hanzi import non_stops
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM = False
if RANDOM:
    random.seed(int(time.time()))

jieba.setLogLevel(logging.INFO)
jieba.load_userdict("user_dict.txt")
eng_punctuation = punctuation.replace("?", "").replace("!", "")
chn_punctuation = non_stops + "｡。" + "→" + "￣" + "．" + " " + "∩" + "°" + "丨" + "´" + "🇳" + "≦" + "🇨" + "≧"


def tokenize(text):
    tokens = list(jieba.cut(Converter("zh-hans").convert(text)))
    processed_tokens = []
    for token in tokens:
        token = token.replace("!", "！").replace("?", "？")
        tmp = token
        for ch in tmp:
            if ch in eng_punctuation or ch in chn_punctuation:
                token = token.replace(ch, "")
        if token != "" and token != " " and token != "" and not token.encode("utf-8").isalnum():
            processed_tokens.append(token)
    if len(processed_tokens) == 0:
        processed_tokens = ["转发微博"]
    return " ".join(processed_tokens)


def split_train_and_test(data_dir):
    # remove former files
    for filename in os.listdir("train"):
        os.remove(os.path.join("train", filename))

    for filename in os.listdir("test"):
        os.remove(os.path.join("test", filename))

    filenames = os.listdir(data_dir)
    for i in range(int(len(filenames) * 0.8)):
        filename = random.choice(filenames)
        shutil.copy(os.path.join(data_dir, filename), os.path.join("train", filename))
        filenames.remove(filename)
    for filename in filenames:
        shutil.copy(os.path.join(data_dir, filename), os.path.join("test", filename))


def generate_tfidf_rumor_model(input_size, data_dir):
    split_train_and_test(data_dir)

    token_dict = {}

    corpus = []
    train_files = os.listdir("train")
    for filename in os.listdir(data_dir):
        if filename in train_files:  # 只取训练集中的前5000个词
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    text = item["text"]
                    tokens = list(jieba.cut(Converter("zh-hans").convert(text)))
                    processed_tokens = []
                    for token in tokens:
                        token = token.replace("!", "！").replace("?", "？")
                        tmp = token
                        for ch in tmp:
                            if ch in eng_punctuation or ch in chn_punctuation:
                                token = token.replace(ch, "")
                        if token != "" and token != " " and token != "" and not token.encode("utf-8").isalnum():
                            processed_tokens.append(token)
                            if token in token_dict:
                                token_dict[token] += 1
                            else:
                                token_dict[token] = 1
                    if len(processed_tokens) == 0:
                        processed_tokens = ["转发微博"]
                        if "转发微博" in token_dict:
                            token_dict["转发微博"] += 1
                        else:
                            token_dict["转发微博"] = 1
                    corpus.append(" ".join(processed_tokens))

    token_list = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)[:input_size]
    vocabulary = {}
    for index, v in enumerate(token_list):
        vocabulary[v[0]] = index

    with open("token_freq.txt", "w", encoding="utf-8", newline="") as f:
        for item in token_list:
            f.write(item[0] + " " + str(item[1]) + "\n")

    tfidf = TfidfVectorizer(vocabulary=vocabulary)
    model = tfidf.fit(corpus)

    return model
