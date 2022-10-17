import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.ap import calculate
from tqdm import tqdm

def valid_cls(args, model, sk_valid_data, im_valid_data):
    model.eval()
    torch.set_grad_enabled(False)

    # if args.retrieval == 'sa':
    #     args.test_sk = 2*args.test_sk
    #     args.test_im = 2*args.test_im

    # if args.train_test_on:
    sk_dataload = DataLoader(sk_valid_data, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
    im_dataload = DataLoader(im_valid_data, batch_size=args.test_im, num_workers=args.num_workers, drop_last=False)

    # print(sk_valid_data[0])
    # print(im_valid_data[-1])

    # print(len(sk_dataload))
    # print(len(im_dataload))

    print('loading image data')
    print('loading sketch data')

    dist_im = None
    all_dist = None
    for i, (sk, sk_label) in enumerate(tqdm(sk_dataload)):

        if i == 0:
            all_sk_label = sk_label.numpy()
        else:
            all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)

        sk_len = sk.size(0)
        sk = sk.cuda()
        sk, sk_idxs = model(sk, None, 'test', only_sa=True)
        # sk, sk_idxs = model(sk, None, 'test', only_sa=True, flag=True)

        for j, (im, im_label) in enumerate(tqdm(im_dataload)):
            if i == 0 and j == 0:
                all_im_label = im_label.numpy()
            elif i == 0 and j > 0:
                all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)

            im_len = im.size(0)
            im = im.cuda()
            im, im_idxs = model(im, None, 'test', only_sa=True)
            # im, im_idxs = model(im, None, 'test', only_sa=True, flag=False)

            # if args.train_test_on:
            sk_temp = sk.unsqueeze(1).repeat(1, im_len, 1, 1).flatten(0, 1).cuda()
            im = im.unsqueeze(0).repeat(sk_len, 1, 1, 1).flatten(0, 1).cuda()

            if args.retrieval == 'rn':
                feature_1, feature_2 = model(sk_temp, im, 'test')
            elif args.retrieval == 'ca':
                feature_1, feature_2 = model(sk_temp, im, 'test')
            elif args.retrieval == 'sa':
                feature_1, feature_2 = torch.cat((sk_temp[:, 0], im[:, 0]), dim=0), None

            # print(feature_1.size())    # [2*sk*im, 768]
            # print(feature_2.size())    # [sk*im, 1]

            # if args.train_test_on:
            if args.retrieval == 'rn':
                if j == 0:
                    dist_im = - feature_2.view(sk_len, im_len).cpu().data.numpy()  # 1*args.batch
                else:
                    dist_im = np.concatenate(
                        (dist_im, - feature_2.view(sk_len, im_len).cpu().data.numpy()), axis=1)
            else:
                temp_dist = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
                                                F.normalize(feature_1[sk_len * im_len:]), 2)
                if j == 0:
                    dist_im = temp_dist.view(sk_len, im_len).cpu().data.numpy()
                else:
                    dist_im = np.concatenate(
                        (dist_im, temp_dist.view(sk_len, im_len).cpu().data.numpy()), axis=1)

        if i == 0:
            all_dist = dist_im
        else:
            all_dist = np.concatenate((all_dist, dist_im), axis=0)

    # print(all_sk_label.size, all_im_label.size)     # [762 x 1711] / 2
    class_same = (np.expand_dims(all_sk_label, axis=1) == np.expand_dims(all_im_label, axis=0)) * 1
    # print(all_dist.size, class_same.size)     # [762 x 1711] / 2
    map_all, precision100, precision200 = calculate(all_dist, class_same, test=True)

    return map_all, precision100, precision200