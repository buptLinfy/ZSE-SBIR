import os
import time
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import cv2


def get_all_train_file(args, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"

        cname_cid = args.data_path + f'/Sketchy/{shot_dir}/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'tu_berlin':
        cname_cid = args.data_path + '/TUBerlin/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'Quickdraw':
        cname_cid = args.data_path + '/QuickDraw/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/sketch_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/all_photo_train.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(skim + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()

    # 图片相对路径
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    # 图片的label,0,1,2...
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    # 所有的训练类
    with open(cname_cid, 'r') as ci:
        class_and_indx = ci.readlines()
    # 类名
    cname = np.array([' '.join(cc.strip().split()[:-1]) for cc in class_and_indx])

    return file_ls, labels, cname


def get_some_file_iccv(labels, rootpath, class_list, cname, number, file_ls):
    file_list = []
    for i in class_list:
        # 该类的label
        label = np.argwhere(cname == i)[0, 0]
        # 该类的所有样本
        ind = np.argwhere(labels == label)
        ind_rand = np.random.randint(1, len(ind), number)
        ind_ori = [ind[i] for i in ind_rand]
        files = [file_ls[i] for i in ind_ori]
        full_path = np.array([os.path.join(rootpath, f[0]) for f in files])
        file_list.append(full_path)
    return file_list


def get_file_iccv(labels, rootpath, class_name, cname, number, file_ls):
    # 该类的label
    label = np.argwhere(cname == class_name)[0, 0]
    # 该类的所有样本
    ind = np.argwhere(labels == label)
    ind_rand = np.random.randint(1, len(ind), number)
    ind_ori = ind[ind_rand]
    files = file_ls[ind_ori][0][0]
    full_path = os.path.join(rootpath, files)
    return full_path


def get_file_list_iccv(args, rootpath, skim, split):

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"
        else:
            NameError("zeroshot is invalid")

        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'

    elif args.dataset == 'tu_berlin':
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'Quickdraw':
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/sketch_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/all_photo_zero.txt'

    else:
        NameError(args.dataset + 'is invalid')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
    file_names = np.array([(rootpath + x) for x in file_ls])

    # 对验证的样本数量进行缩减
    # sketch 15229->762 image 17101->1711
    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'sketch':
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 20*args.split)]   # 762
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 15229
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'images':
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 10*args.split)]  # 1711
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 17101
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # sketch 2400->800, image 27989->1400
    if args.dataset == "tu_berlin" and skim == "sketch" and split == "test":
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 3*args.split)]  # 800
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 2400
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "tu_berlin" and skim == "images" and split == "test":
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 20*args.split)]   # 1400
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 27989
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # Quickdraw 92291->770, image 54151->1806
    if args.dataset == "Quickdraw" and skim == "sketch" and split == "test":
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 120*args.split)]  # 770
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 92291
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "Quickdraw" and skim == "images" and split == "test":
        if args.split:
            index = [i for i in range(0, file_names.shape[0], 30*args.split)]  # 1806
        else:
            index = [i for i in range(0, file_names.shape[0], 1)]   # 54151
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    file_names_cls = labels
    return file_names, file_names_cls



def preprocess(image_path, img_type="im"):
    # immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]

    immean = [0.5, 0.5, 0.5]  # RGB channel mean for imagenet
    imstd = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(immean, imstd)
    ])

    if img_type == 'im':
        return transform(Image.open(image_path).resize((224, 224)).convert('RGB'))
    else:
        # 对sketch 进行crop，等比例扩大到224
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_white_space_image(img, 10)
        img = resize_image_by_ratio(img, 224)
        img = make_img_square(img)

        return transform(img)


def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    # if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
    #     img_np = (img_np * 255).astype("uint8")
    # else:
    #     img_np = img_np.astype("uint8")

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped


def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    :param img_np:
    :param size:
    :return:
    """
    # print(len(img_np.shape))
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    # new_img[np.where(new_img < 200)] = 0
    return new_img


def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((h, delta2, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((delta2, w, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img


# 每个label，对应一个数字
def create_dict_texts(texts):
    texts = list(texts)
    dicts = {l: i for i, l in enumerate(texts)}
    return dicts
