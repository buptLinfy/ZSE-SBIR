import os
import sys
import shutil
import colorsys

import torch
import torch.nn.functional as F

def get_choose_token(idxs):

    v9 = idxs[9][0]
    v6 = idxs[6][0]
    v3 = idxs[3][0]
    # print(v9)
    # print(v6[v9])
    # print(v3[v6[v9]])

    choose_token = v3[v6[v9]].cpu().numpy().tolist()
    return choose_token


def mask_choose_token(choose_token, black=False):
    w_featmap = 14
    h_featmap = 14
    patch_size = 16

    if black == False:
        mask = torch.zeros((1, 196))
        mask[0, choose_token] = 1
    else:
        mask = torch.ones((1, 196))
        mask[0, choose_token] = 0

    mask = mask.reshape(1, w_featmap, h_featmap).float()
    # print('choose_token_mask:', mask.shape)     # 1, 14, 14
    mask = F.interpolate(mask.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].numpy()
    # print('choose_token_mask:', mask.shape)     # 1, 224, 224
    return mask


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        # image[:, :, c] = image[:, :, c] + alpha * mask * color[c] * 255
        # print(image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

