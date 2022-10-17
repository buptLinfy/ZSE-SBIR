import copy
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data
from model.model import Model
from utils.util import build_optimizer, save_checkpoint, setup_seed, load_checkpoint
from utils.ap import calculate
from loss.loss import triplet_loss, rn_loss

from tqdm import tqdm

from valid import valid_cls


def train():
    train_data, sk_valid_data, im_valid_data = load_data(args)

    model = Model(args.d_model, args)

    # batch=15, lr=1e-5 / batch=30, lr=2e-5
    optimizer = build_optimizer(args, model)

    if args.number_gpu > 1:
        # model = torch.nn.DataParallel(model, device_ids=list(range(args.number_gpu)))
        model = torch.nn.parallel.DataParallel(model.to('cuda'))
    model = model.cuda()

    train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)

    start_epoch = 0
    accuracy = 0
    precision = 0

    for i in range(start_epoch, args.epoch):
        print('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)

        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch

        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, sk_label_n, im_label_n) in enumerate(train_data_loader):

            sk = torch.cat((sk, sk_neg))
            im = torch.cat((im, im_neg))
            sk, im = sk.cuda(), im.cuda()

            cls_fea, rn_scores = model(sk, im)

            # if rn_scores is not None:
            target_rn = torch.cat((torch.ones(sk_label.size()), torch.zeros(sk_label.size())), dim=0)
            target_rn = torch.clamp(target_rn, 0.01, 0.99).unsqueeze(dim=1)
            target_rn = target_rn.cuda()

            losstri = triplet_loss(cls_fea, args) * 2
            lossrn = rn_loss(rn_scores, target_rn) * 4  # loss2 初始值应为1左右

            loss = losstri + lossrn

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f} '
                      f'tri:{losstri.item():.3f} rn:{lossrn.item():.3f}')

        if epoch >= 5:
            print('------------------------valid------------------------')
            map_all, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
            print(f'map_all:{map_all:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')

            print("Save the {}th model......".format(epoch))
            save_checkpoint(
                {'model': model.state_dict(), 'epoch': epoch, 'map_all': map_all, 'precision_100': precision_100},
                args.save, f'checkpoint_{epoch}_mapall_{map_all:.4f}_prec100_{precision_100:.4f}')

            if map_all > accuracy and precision_100 > precision:
                accuracy = map_all
                precision = precision_100

                print("Best the {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch, 'map_all': accuracy, 'precision_100': precision},
                    args.save, f'best_checkpoint')

        # if epoch % 5 == 0:
        #     print("Regular Save the {}th model......".format(epoch))
        #     save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'accuracy': map_all},
        #                     args.save, f'{epoch}_checkpoint_mapall_{map_all:.4f}')


if __name__ == '__main__':
    args = Option().parse()
    print(str(args))

    # if args.number_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)

    setup_seed(args.seed)

    writer = SummaryWriter(args.tensorboard)

    train()
