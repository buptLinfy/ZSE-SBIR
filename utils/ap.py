import multiprocessing
import time
import numpy as np
from joblib import delayed, Parallel


def calculate(distance, class_same, dict_class=None, acc_cls_sk=None, test=None):
    arg_sort_sim = distance.argsort()  # 得到从小到大【索引值】
    sort_label = []
    for index in range(0, arg_sort_sim.shape[0]):
        # 将label重新排序，根据距离的远近，距离越近的排在前面
        sort_label.append(class_same[index, arg_sort_sim[index, :]])
    sort_label = np.array(sort_label)
    # print(arg_sort_sim)
    # print(sort_label)
    # 多进程计算
    num_cores = min(multiprocessing.cpu_count(), 4)

    if test:
        # 测试时使用voc_eval 结果会高0.07左右，但是速度慢
        start = time.time()
        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        aps_200 = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq], 200) for iq in range(distance.shape[0]))
        aps_100 = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq], 100) for iq in range(distance.shape[0]))

        # if dict_class is not None:
        #     dict_class = {v: k for k, v in dict_class.items()}
        #     diff_class = set(acc_cls_sk)
        #     for cls in diff_class:
        #         ind = acc_cls_sk == cls
        #         each_cls_map = np.array(aps_all)[ind].mean()
        #         print('mAP_all {} class {}'.format(str(each_cls_map), dict_class[cls]))

        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))
        precision_200 = np.nanmean(precision_200)
        precision_100 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 100) for iq in range(sort_label.shape[0]))
        precision_100 = np.nanmean(precision_100)

        aps_all = np.nanmean(aps_all)
        print("eval time:", time.time() - start)
        return aps_all, precision_100, precision_200

    else:
        # 训练时使用othermap1，是因为速度快，但是结果稍逊0.07
        start = time.time()
        aps_all = Parallel(n_jobs=num_cores)(
            delayed(other_map1)(sort_label[iq]) for iq in range(distance.shape[0]))

        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))

        precision_200 = np.nanmean(precision_200)

        precision_100 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 100) for iq in range(sort_label.shape[0]))

        precision_100 = np.nanmean(precision_100)

        aps_all = np.nanmean(aps_all)
        print("eval time:", time.time() - start)
        return aps_all, precision_100, precision_200


def voc_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    fp = np.logical_not(tp)
    tot = tp.shape[0]
    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        precision = tp / (tp + fp)
    except:
        print("error", tot_pos)
        return np.nan

    ap = voc_ap(rec, precision)
    return ap


def precision_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    # tot = tp.shape[0]

    if top is not None:
        top = min(top, tot_pos)
    else:
        top = tot_pos

    return np.mean(sort_class_same[:top])


def other_map1(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    tot = tp.shape[0]

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        tot_pos = min(top, tot_pos)

    ap_sum = 0
    number = 0
    for i in range(len(tp)):
        if tp[i]:
            number += 1
            ap_sum += number / (i + 1)
            if number == tot_pos:
                break
    # print(ap_sum,tot_pos)
    ap = ap_sum / (tot_pos + 1e-5)
    return ap


def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap
