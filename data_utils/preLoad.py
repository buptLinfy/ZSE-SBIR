from .utils import get_file_list_iccv, get_all_train_file
import numpy as np


# 预加载的一些文件
def load_para(args):
    # test class labels
    if args.dataset == 'sketchy_extend':
        if args.test_class == 'test_class_sketchy25':

            with open(args.data_path + "/Sketchy/zeroshot1/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

            train_dir = args.data_path + "/Sketchy/zeroshot1/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

        elif args.test_class == "test_class_sketchy21":  # 21个类
            with open(args.data_path + "/Sketchy/zeroshot0/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/Sketchy/zeroshot0/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        elif args.test_class == "test_class_sketchyfew":
            with open(args.data_path + "/Sketchy/" + args.zeroversion + "/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/Sketchy/" + args.zeroversion + "/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.dataset == 'tu_berlin':
        if args.test_class == 'test_class_tuberlin30':
            with open(args.data_path + "/TUBerlin/" + args.zeroversion + "/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/TUBerlin/" + args.zeroversion + "/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    elif args.dataset == 'Quickdraw':
        with open(args.data_path + "/QuickDraw/zeroshot/cname_cid_zero.txt", 'r') as f:
            file_content = f.readlines()
            test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        train_dir = args.data_path + "/QuickDraw/zeroshot/cname_cid.txt"
        with open(train_dir, 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    print('training classes: ', train_class_label.shape)
    print('testing classes: ', test_class_label.shape)
    return train_class_label, test_class_label


class PreLoad:
    def __init__(self, args):
        self.all_valid_or_test_sketch = []
        self.all_valid_or_test_sketch_label = []
        self.all_valid_or_test_image = []
        self.all_valid_or_test_image_label = []

        self.all_train_sketch = []
        self.all_train_sketch_label = []
        self.all_train_image = []
        self.all_train_image_label = []

        self.all_train_sketch_cls_name = []
        self.all_train_image_cls_name = []

        self.init_valid_or_test(args)
        # load_para(args)

    def init_valid_or_test(self, args):
        if args.dataset == 'sketchy_extend':
            train_dir = args.data_path + '/Sketchy/'
        elif args.dataset == 'tu_berlin':
            train_dir = args.data_path + '/TUBerlin/'
        elif args.dataset == 'Quickdraw':
            train_dir = args.data_path + '/QuickDraw/'
        else:
            NameError("Dataset is not implemented")

        if args.stage == "train":
            split = "val"
        elif args.stage == "test":
            split = "test"
            print("args.stage ---->  test.........")
        else:
            NameError("stage is not right")

        self.all_valid_or_test_sketch, self.all_valid_or_test_sketch_label = \
            get_file_list_iccv(args, train_dir, "sketch", "test")
        self.all_valid_or_test_image, self.all_valid_or_test_image_label = \
            get_file_list_iccv(args, train_dir, "images", "test")

        self.all_train_sketch, self.all_train_sketch_label, self.all_train_sketch_cls_name =\
            get_all_train_file(args, "sketch")
        self.all_train_image, self.all_train_image_label, self.all_train_image_cls_name = \
            get_all_train_file(args, "image")

        # print(len(self.all_train_image_label))
        print("used for valid or test sketch / image:")
        print(self.all_valid_or_test_sketch.shape, self.all_valid_or_test_image.shape)
        print("used for train sketch / image:")
        print(self.all_train_sketch.shape, self.all_train_image.shape)
