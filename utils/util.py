import errno
import os
import sys
import shutil
import torch
import random
import numpy as np

from torch.optim import AdamW


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def l1_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.abs(x - y).sum(2)


def build_optimizer(args, model):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def build_optimizer_finetune(args, model):

    # # 工作1e-5
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    model_lr = {'rn': 2e-5, 'others': 1e-5}
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    for layer_name in model_lr:
        lr = model_lr[layer_name]
        if layer_name == 'rn':  # 设定了特定 lr 的 layer
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": 1e-2,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and layer_name in n)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        else:  # 其他，默认学习率
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": 1e-2,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    return optimizer


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(root_save_path):
    if os.path.exists(root_save_path):
        shutil.rmtree(root_save_path)  # delete output folder
    os.makedirs(root_save_path)  # make new output folder


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass