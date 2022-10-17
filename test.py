import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data_test
from model.model import Model
from utils.util import save_checkpoint, setup_seed, load_checkpoint
from utils.ap import calculate
from loss.loss import triplet_loss, rn_loss

from tqdm import tqdm

from valid import valid_cls

def test():
    sk_valid_data, im_valid_data = load_data_test(args)

    model = Model(args.d_model, args)
    #half
    model = model.half()

    # if args.load is not None:
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0716_ori_196/checkpoint_12_mapall_0.7248.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0716_ori_196/checkpoint_16_mapall_0.7310.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0728_ori_196_30/checkpoint_18_mapall_0.7516_prec100_0.7063.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0729_0.7_68/checkpoint_12_mapall_0.6700_prec100_0.6262.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0715_0.9_144_finetune_diff_sort_1/checkpoint_6_mapall_0.6790.pth")
    checkpoint = load_checkpoint("/root/lfy/HANet/save/0716_0.7_68_sort_15/checkpoint_18_mapall_0.6570.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0729_0.7_68/checkpoint_13_mapall_0.6531_prec100_0.6112.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0730_0.7_68/checkpoint_12_mapall_0.6557_prec100_0.6147.pth")
    # checkpoint = load_checkpoint("/root/lfy/HANet/save/0717_0.9_144_sort_15/checkpoint_14_mapall_0.6709.pth")

    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)

    if len(args.choose_cuda) > 1:
        model = torch.nn.parallel.DataParallel(model.to('cuda'))
    model = model.cuda()

    map_all, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
    print(f'map_all:{map_all:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')


if __name__ == '__main__':
    args = Option().parse()
    print(str(args))

    # if args.number_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)

    setup_seed(args.seed)

    writer = SummaryWriter(args.tensorboard)

    test()
