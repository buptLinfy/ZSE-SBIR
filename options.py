import argparse


class Option:

    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset
        parser.add_argument('--dataset', type=str, default='sketchy_extend',
                            choices=['sketchy_extend', 'tu_berlin', 'Quickdraw', 'ShoeV2', 'ChairV2'])
        parser.add_argument('--data_path', type=str, default="/root/project/SAKE/dataset")
        # parser.add_argument('--data_path', type=str, default="/root/data/patent_data")

        parser.add_argument('--test_class', type=str,
                            default='test_class_sketchy25',
                            choices=['test_class_sketchy25', 'test_class_sketchyfew', 'test_class_sketchy21',
                                     'test_class_tuberlin30', 'Quickdraw', 'ChairV2', 'ShoeV2'])
        parser.add_argument('--zeroversion', type=str, default='zeroshot')

        parser.add_argument('--d_model', type=int, default=768)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--h', default=8, type=int)
        parser.add_argument('--number', default=1, type=int, help='the number of stack encoderLayer')
        parser.add_argument('--epoch', default=20, type=int)

        parser.add_argument('--weight_decay', default=1e-2, type=float)
        parser.add_argument('--batch', type=int, default=30)
        parser.add_argument('--pretrained', default=True, action='store_false', help='default:True')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--max_norm', type=float, default=0.1)
        parser.add_argument('--datasetLen', type=int, default=10000)

        # parser.add_argument('--pd_learn', action='store_true', help='位置信息学习')
        parser.add_argument('--anchor_number', '-a', type=int, default=196)
        parser.add_argument('--patch_size', type=int, default=14, help='patch size')
        parser.add_argument('--patch_number_sqrt', type=int, default=16)
        parser.add_argument('--cls_number', type=int, default=100)

        # test
        parser.add_argument('--train_test_on', action='store_false', help='训练阶段的valid输入大小控制，控制下方2参数')
        parser.add_argument('--test_sk', type=int, default=30)
        parser.add_argument('--test_im', type=int, default=30)

        # other
        parser.add_argument('--number_gpu', '-n', type=int, default=1, choices=[0,1,2],
                            help='0 = CPU, 1 = CUDA, 2 = DataParallel')
        parser.add_argument('--choose_cuda', '-c', type=str, default='0')
        parser.add_argument('--stage', type=str, default='train')
        parser.add_argument('--tensorboard', '-t', type=str, default='/root/tensorboard/log')
        parser.add_argument('--save', '-s', type=str, default='./checkpoints/test01', help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint File')

        # loss
        parser.add_argument('--margin', type=float, default=1.0, help='The distance of adv-n - adv-p')
        parser.add_argument('--p', type=int, default=2, help='distance for tripletMarginLoss')
        parser.add_argument('--triplet', default=True, action='store_false', help='default:True')

        parser.add_argument('--split', type=int, default=1, help='train/test scale')
        parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
        parser.add_argument("--seed", type=int, default=2021, help="random seed.")

        parser.add_argument('--retrieval', '-r', type=str, default='rn', choices=['rn', 'ca', 'sa'])

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
