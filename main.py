import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import json
from Tree import Tree
from data_preprocess import generate_tfidf_rumor_model
import os
import jieba
import logging
import random
import time
from langconv import *
from zhon.hanzi import non_stops
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from MultiModel import MultiModel

random.seed(int(time.time()))
jieba.setLogLevel(logging.INFO)
jieba.load_userdict("user_dict.txt")
eng_punctuation = punctuation.replace("?", "").replace("!", "")
chn_punctuation = non_stops + "｡。" + "→" + "￣" + "．" + " " + "∩" + "°" + "丨" + "´" + "🇳" + "≦" + "🇨" + "≧"


def get_batch_rumor(batch_size, filenames):
    random.shuffle(train_data_files)
    start_index = 0
    end_index = batch_size
    while end_index < len(filenames):
        batch = load_trees("train", filenames[start_index: end_index])
        temp = end_index
        end_index = end_index + batch_size
        start_index = temp
        yield batch

    if end_index >= len(filenames):
        batch = load_trees("train", filenames[start_index:])
        yield batch



def get_batch_stance(batch_size, data):
    for X, labels in data:
        title_vector = X[0]
        start_index = 1
        end_index = batch_size + 1
        while end_index < len(X):
            batch_X = torch.cat([FloatTensor(title_vector.reshape(1, -1)), FloatTensor(X[start_index: end_index, :])], dim=0)
            batch_labels = labels[start_index - 1: end_index - 1]
            temp = end_index
            end_index = end_index + batch_size
            start_index = temp
            yield batch_X, batch_labels

        if end_index >= len(X):
            batch_X = torch.cat([FloatTensor(title_vector.reshape(1, -1)), FloatTensor(X[start_index:, :])], dim=0)
            batch_labels = labels[start_index - 1:]
            yield batch_X, batch_labels


def load_trees(dataset, filenames):
    trees = []
    for filename in filenames:
        trees.append(Tree(os.path.join(dataset, filename), tfidf_rumor_model, INPUT_SIZE_RUMOR))
    return trees


def load_stances(input_size):
    ret = []
    for filename in os.listdir("stance"):
        corpus = [filename.rstrip(".txt")]
        labels = []
        token_dict = {}
        with open(os.path.join("stance", filename), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                text = line.split("\t")[2]
                label = line.split("\t")[3]
                if label == "FAVOR":
                    label = 0
                elif label == "AGAINST":
                    label = 1
                else:
                    label = 2
                labels.append(label)

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

                corpus.append(" ".join(processed_tokens))

        token_list = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)[:input_size]
        vocabulary = {}
        for index, v in enumerate(token_list):
            vocabulary[v[0]] = index
        tfidf = TfidfVectorizer(vocabulary=vocabulary)
        X = tfidf.fit_transform(corpus).toarray()
        ret.append((X, labels))
    return ret


USE_CUDA = torch.cuda.is_available()
gpus = [3]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

INPUT_SIZE_RUMOR = 2000
INPUT_SIZE_STANCE = 2000
RUMOR_CLASSES = 2
STANCE_CLASSES = 3
HIDDEN_SIZE = 100
EPOCH = 600
LR = 0.005
LAMBDA = 0
BATCH_SIZE_RUMOR = 1
BATCH_SIZE_STANCE = 10

if USE_CUDA:
    print("GPU Mode.")
else:
    print("CPU Mode.")

tfidf_rumor_model = generate_tfidf_rumor_model(input_size=INPUT_SIZE_RUMOR, data_dir="data_all")
print("Split train-test sets / Load tf-idf model finished.")

train_data_files = os.listdir("train")
test_data_files = os.listdir("test")
test_data = load_trees("test", test_data_files)
stance_data = load_stances(INPUT_SIZE_STANCE)
print("Load data finished.")

model = MultiModel(INPUT_SIZE_RUMOR, INPUT_SIZE_STANCE, HIDDEN_SIZE, RUMOR_CLASSES, STANCE_CLASSES)
model.init_weight()

if USE_CUDA:
    model = model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=LAMBDA)
print("Model init finished.")


for epoch in range(EPOCH):
    losses_rumor = []
    losses_stance = []
    # train
    model.train()
    rumor_iter = get_batch_rumor(BATCH_SIZE_RUMOR, train_data_files)
    stance_iter = get_batch_stance(BATCH_SIZE_STANCE, stance_data)

    # input_data_train = train_data.copy()

    REMAIN_RUMOR = True
    REMAIN_STANCE = True

    while REMAIN_RUMOR or REMAIN_STANCE:
        # choose a task randomly
        if REMAIN_RUMOR and not REMAIN_STANCE:
            task = 1
        elif not REMAIN_RUMOR and REMAIN_STANCE:
            task = 2
        else:
            task = random.randint(1, 2)

        if task == 1:
            try:
                batch = next(rumor_iter)
            except StopIteration:
                REMAIN_RUMOR = False
                continue
            model.zero_grad()
            labels = LongTensor([tree.label for tree in batch])
            preds = model("rumor", batch)

            loss = loss_function(preds, labels)
            losses_rumor.append(loss.data.tolist())
            loss.backward()
            optimizer.step()
        else:
            try:
                batch_X, batch_labels = next(stance_iter)
            except StopIteration:
                REMAIN_STANCE = False
                continue
            model.zero_grad()
            labels = LongTensor(batch_labels)
            preds = model("stance", batch_X)

            loss = loss_function(preds, labels)
            losses_stance.append(loss.data.tolist())
            loss.backward()
            optimizer.step()

    print("[%d/%d] rumor mean_loss: %.4f stance mean_loss: %.4f" % (epoch + 1, EPOCH, float(np.mean(losses_rumor)), float(np.mean(losses_stance))))
    torch.save(model.state_dict(), os.path.join("model", "checkpoint_%d.pth" % epoch))


    # test
    # class 0: non-rumor class 1: rumor
    # if (epoch + 1) % 5 == 0:
    model.eval()

    accuracy = 0
    total_samples = 0
    TP0, FP0, FN0, TN0, F1_0 = 0, 0, 0, 0, 0
    TP1, FP1, FN1, TN1, F1_1 = 0, 0, 0, 0, 0
    model.zero_grad()
    preds = model("rumor", test_data)
    labels = [tree.label for tree in test_data]
    for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
        total_samples += 1
        if pred == 0 and label == 0:
            TP0 += 1
            TN1 += 1
            accuracy += 1
        if pred == 0 and label == 1:
            FN0 += 1
            FP1 += 1
        if pred == 1 and label == 0:
            FP0 += 1
            FN1 += 1
        if pred == 1 and label == 1:
            TN0 += 1
            TP1 += 1
            accuracy += 1
    pre0 = TP0 / (TP0 + FP0)  # precision
    rec0 = TP0 / (TP0 + FN0)  # recall
    pre1 = TP1 / (TP1 + FP1)
    rec1 = TP1 / (TP1 + FN1)
    f1_0 = 2 * pre0 * rec0 / (pre0 + rec0)  # f1
    f1_1 = 2 * pre1 * rec1 / (pre1 + rec1)
    print("Accuracy: %.4f%%" % (accuracy / total_samples * 100))
    print("Non-rumor precision: %.4f%% recall: %.4f%% f1: %.4f" % (pre0 * 100, rec0 * 100, f1_0))
    print("Rumor precision: %.4f%% recall: %.4f%% f1: %.4f" % (pre1 * 100, rec1 * 100, f1_1))




