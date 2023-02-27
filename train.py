import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data
from model.model import Model
from utils.util import build_optimizer, save_checkpoint, setup_seed
from utils.loss import triplet_loss, rn_loss
from utils.valid import valid_cls


def train():
    train_data, sk_valid_data, im_valid_data = load_data(args)

    model = Model(args)
    model = model.cuda()

    # batch=15, lr=1e-5 / batch=30, lr=2e-5
    optimizer = build_optimizer(args, model)

    train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)

    start_epoch = 0
    accuracy = 0

    for i in range(start_epoch, args.epoch):
        print('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)

        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch

        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(tqdm(train_data_loader)):
            # prepare data
            sk = torch.cat((sk, sk_neg))
            im = torch.cat((im, im_neg))
            sk, im = sk.cuda(), im.cuda()

            # prepare rn truth
            target_rn = torch.cat((torch.ones(sk_label.size()), torch.zeros(sk_label.size())), dim=0)
            target_rn = torch.clamp(target_rn, 0.01, 0.99).unsqueeze(dim=1)
            target_rn = target_rn.cuda()

            # calculate feature
            cls_fea, rn_scores = model(sk, im)

            # loss
            losstri = triplet_loss(cls_fea, args) * 2   # losstri 初始值应为1左右
            lossrn = rn_loss(rn_scores, target_rn) * 4  # lossrn  初始值应为1左右
            loss = losstri + lossrn

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f} '
                      f'tri:{losstri.item():.3f} rn:{lossrn.item():.3f}')

        if epoch >= 10:
            print('------------------------valid------------------------')
            # log
            map_all, map_200, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
            print(f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')
            # save
            if map_all > accuracy:
                accuracy = map_all
                precision = precision_100
                print("Save the BEST {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch, 'map_all': accuracy, 'precision_100': precision},
                    args.save, f'best_checkpoint')


if __name__ == '__main__':
    args = Option().parse()
    print("train args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    train()
