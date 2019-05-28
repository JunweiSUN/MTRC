import torch
import json
from data_preprocess import tokenize

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class Node:
    def __init__(self):
        self.word_vec = None
        self.parent = None
        self.is_leaf = False
        self.children = None


class Tree:
    def __init__(self, filepath, tfidf_model, input_size):
        self.label = int(filepath.strip(".json").split("_")[1])
        self.post_dict = {} # mid : post
        self.parent_dict = {} # node_mid : parent_mid
        self.children_dict = {} # node_mid : [child1_mid, child2_mid, ...]
        self.root_mid = None
        self.parse(filepath)
        self.input_size = input_size
        self.root = self.generate_tree(tfidf_model, self.root_mid, None)

    def parse(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            event = json.load(f)
            for post in event:
                mid = post["mid"]
                parent = post["parent"]
                if parent is None:
                    self.parent_dict[mid] = None
                    self.root_mid = mid
                else:
                    self.parent_dict[mid] = parent
                    if parent not in self.children_dict.keys():
                        self.children_dict[parent] = []
                    self.children_dict[parent].append(mid)
                self.post_dict[mid] = post

    def generate_tree(self, tfidf_model, mid, parent=None):
        node = Node()
        post = self.post_dict[mid]
        node.word_vec = FloatTensor(tfidf_model.transform([tokenize(post["text"])]).toarray().reshape(self.input_size, -1))
        node.parent = parent

        # leaf node
        if mid not in self.children_dict.keys():
            node.is_leaf = True
            return node

        # non-leaf node
        node.children = []
        for child_mid in self.children_dict[mid]:
            node.children.append(self.generate_tree(tfidf_model, child_mid, node))

        return node
