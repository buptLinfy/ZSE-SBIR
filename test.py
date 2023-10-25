import os

import torch

from options import Option
from data_utils.dataset import load_data_test
from model.model import Model
from utils.util import setup_seed, load_checkpoint

from utils.valid import valid_cls

def test():
    # prepare data
    sk_valid_data, im_valid_data = load_data_test(args)

    # prepare model
    model = Model(args)
    model = model.half()

    if args.load is not None:
        checkpoint = load_checkpoint(args.load)

    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)

    if len(args.choose_cuda) > 1:
        model = torch.nn.parallel.DataParallel(model.to('cuda'))
    model = model.cuda()

    # valid
    map_all, map_200, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
    print(f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')


if __name__ == '__main__':
    args = Option().parse()
    print("test args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    test()
