import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import json

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class MultiModel(nn.Module):
    def __init__(self, input_size_rumor, input_size_stance, hidden_size, rumor_classes, stance_classes):
        super(MultiModel, self).__init__()
        self.input_size_rumor = input_size_rumor
        self.input_size_stance = input_size_stance
        self.hidden_size = hidden_size
        self.rumor_classes = rumor_classes
        self.stance_classes = stance_classes

        # rumor specific layer
        self.Er = nn.Parameter(torch.randn(hidden_size, input_size_rumor))
        self.Wr_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ur_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2r_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wr_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ur_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2r_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wr_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ur_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2r_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Vr = nn.Parameter(torch.randn(rumor_classes, hidden_size))
        self.br = nn.Parameter(torch.zeros(rumor_classes, 1))

        # stance specific layer
        self.Es = nn.Parameter(torch.randn(hidden_size, input_size_stance))
        self.Ws_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Us_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2s_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ws_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Us_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2s_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ws_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Us_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr2s_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Vs_1 = nn.Parameter(torch.randn(stance_classes, hidden_size))
        self.Vs = nn.Parameter(torch.randn(stance_classes, hidden_size))
        self.bs = nn.Parameter(torch.zeros(stance_classes, 1))

        # shared layer
        self.Wsr_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsr_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsr_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Usr_h = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def init_weight(self):
        nn.init.xavier_uniform_(self.Er)
        nn.init.xavier_uniform_(self.Wr_r)
        nn.init.xavier_uniform_(self.Ur_r)
        nn.init.xavier_uniform_(self.Usr2r_r)
        nn.init.xavier_uniform_(self.Wr_z)
        nn.init.xavier_uniform_(self.Ur_z)
        nn.init.xavier_uniform_(self.Usr2r_z)
        nn.init.xavier_uniform_(self.Wr_h)
        nn.init.xavier_uniform_(self.Ur_h)
        nn.init.xavier_uniform_(self.Usr2r_h)
        nn.init.xavier_uniform_(self.Vr)

        nn.init.xavier_uniform_(self.Es)
        nn.init.xavier_uniform_(self.Ws_r)
        nn.init.xavier_uniform_(self.Us_r)
        nn.init.xavier_uniform_(self.Usr2s_r)
        nn.init.xavier_uniform_(self.Ws_z)
        nn.init.xavier_uniform_(self.Us_z)
        nn.init.xavier_uniform_(self.Usr2s_z)
        nn.init.xavier_uniform_(self.Ws_h)
        nn.init.xavier_uniform_(self.Us_h)
        nn.init.xavier_uniform_(self.Usr2s_h)
        nn.init.xavier_uniform_(self.Vs_1)
        nn.init.xavier_uniform_(self.Vs)

        nn.init.xavier_uniform_(self.Wsr_r)
        nn.init.xavier_uniform_(self.Usr_r)
        nn.init.xavier_uniform_(self.Wsr_z)
        nn.init.xavier_uniform_(self.Usr_z)
        nn.init.xavier_uniform_(self.Wsr_h)
        nn.init.xavier_uniform_(self.Usr_h)

    def rumor_forward(self, node, h_shared_prev, h_rumor_prev):
        x = node.word_vec
        x = torch.reshape(x, (self.input_size_rumor, 1))
        x_tilde = torch.mm(self.Er, x)

        r_shared = torch.sigmoid(torch.mm(self.Wsr_r, x_tilde) + torch.mm(self.Usr_r, h_shared_prev))
        z_shared = torch.sigmoid(torch.mm(self.Wsr_z, x_tilde) + torch.mm(self.Usr_z, h_shared_prev))
        h_shared_tilde = torch.tanh(torch.mm(self.Wsr_h, x_tilde) + torch.mm(self.Usr_h, h_shared_prev * r_shared))
        h_shared = ((1 - z_shared) * h_shared_prev) + (z_shared * h_shared_tilde)

        r_rumor = torch.sigmoid(torch.mm(self.Wr_r, x_tilde) + torch.mm(self.Ur_r, h_rumor_prev) + torch.mm(self.Usr2r_r, h_shared))
        z_rumor = torch.sigmoid(torch.mm(self.Wr_z, x_tilde) + torch.mm(self.Ur_z, h_rumor_prev) + torch.mm(self.Usr2r_z, h_shared))
        h_rumor_tilde = torch.tanh(torch.mm(self.Wr_h, x_tilde) + torch.mm(self.Ur_h, h_rumor_prev * r_rumor) + torch.mm(self.Usr2r_h, h_shared))
        h_rumor = ((1 - z_rumor) * h_rumor_prev) + (z_rumor * h_rumor_tilde)

        # leaf node
        if node.is_leaf:
            return [h_rumor]

        # non-leaf node
        ret = []
        for child_node in node.children:
            ret += self.rumor_forward(child_node, h_shared, h_rumor)
        return ret

    def stance_forward(self, x, h_shared_prev, h_stance_prev):
        x = torch.reshape(x, (self.input_size_stance, 1))
        x_tilde = torch.mm(self.Es, x)

        r_shared = torch.sigmoid(torch.mm(self.Wsr_r, x_tilde) + torch.mm(self.Usr_r, h_shared_prev))
        z_shared = torch.sigmoid(torch.mm(self.Wsr_z, x_tilde) + torch.mm(self.Usr_z, h_shared_prev))
        h_shared_tilde = torch.tanh(torch.mm(self.Wsr_h, x_tilde) + torch.mm(self.Usr_h, h_shared_prev * r_shared))
        h_shared = ((1 - z_shared) * h_shared_prev) + (z_shared * h_shared_tilde)

        r_stance = torch.sigmoid(torch.mm(self.Ws_r, x_tilde) + torch.mm(self.Us_r, h_stance_prev) + torch.mm(self.Usr2s_r, h_shared))
        z_stance = torch.sigmoid(torch.mm(self.Ws_z, x_tilde) + torch.mm(self.Us_z, h_stance_prev) + torch.mm(self.Usr2s_z, h_shared))
        h_rumor_tilde = torch.tanh(torch.mm(self.Ws_h, x_tilde) + torch.mm(self.Us_h, h_stance_prev * r_stance) + torch.mm(self.Usr2s_h, h_shared))
        h_stance = ((1 - z_stance) * h_stance_prev) + (z_stance * h_rumor_tilde)

        return h_shared, h_stance

    def forward(self, task, data):  # task: "rumor" or "stance"
        if task == "rumor":
            preds = []
            for tree in data:
                h_shared = torch.zeros(self.hidden_size, 1).cuda()
                h_rumor = torch.zeros(self.hidden_size, 1).cuda()
                leaf_vectors = self.rumor_forward(tree.root, h_shared, h_rumor)
                output_layer_input = torch.max(torch.cat(leaf_vectors, dim=1), dim=1, keepdim=True)[0]
                pred = F.softmax(torch.mm(self.Vr, output_layer_input) + self.br, dim=0)
                preds.append(pred)
            preds = torch.cat(preds, dim=1).t()
            return preds
        else:
            h_shared = torch.zeros(self.hidden_size, 1).cuda()
            h_stance = torch.zeros(self.hidden_size, 1).cuda()
            hs_1 = torch.zeros(self.hidden_size, 1).cuda()
            
            preds = []
            for i, x in enumerate(data):
                h_shared, h_stance = self.stance_forward(x, h_shared, h_stance)
                if i == 0:
                    hs_1.copy_(h_stance)
                else:
                    pred = F.softmax(torch.mm(self.Vs_1, hs_1) + torch.mm(self.Vs, h_stance) + self.bs, dim=0)
                    preds.append(pred)
            preds = torch.cat(preds, dim=1).t()
            return preds
