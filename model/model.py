import torch
import torch.nn as nn
from .sa import Self_Attention
from .ca import Cross_Attention
from .rn import Relation_Network, cos_similar

from options import Option

class Model(nn.Module):
    def __init__(self, d_model, args):
        super(Model, self).__init__()

        self.args = args

        self.sa = Self_Attention(d_model=d_model, cls_number=args.cls_number, pretrained=args.pretrained)  # b*512*7*7
        self.ca = Cross_Attention(args=args, h=args.h, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.rn = Relation_Network(args.anchor_number, dropout=0.1)

        # anchor -> 1
        self.conv2d = nn.Conv2d(768, 512, 2, 2)

        # 降维
        # self.linear = nn.Linear(768, 512)


    def forward(self, sk, im, stage='train', only_sa=False):

        if stage == 'train':

            sk_im = torch.cat((sk, im), dim=0)
            sa_fea, left_tokens, idxs = self.sa(sk_im)  # b*197*768 -> b*69*768

            # print('sa_fea:', sa_fea.size())
            # print('left_tokens:', left_tokens)
            # print('idxs:', idxs[9])

            # transformer
            ca_fea = self.ca(sa_fea, stage)  # b*seq_len*768

            cls_fea = ca_fea[:, 0]
            token_fea = ca_fea[:, 1:]
            batch = token_fea.size(0)

            # print('token_fea:', token_fea.size())  # b*68*768

            # 分类
            # x1 = self.cls(cls_fea)

            token_fea = token_fea.view(batch, 768, 14, 14)
            down_fea = self.conv2d(token_fea)
            down_fea = down_fea.view(batch, 512, 7*7)
            down_fea = down_fea.transpose(1, 2)

            # down_fea = self.linear(token_fea)
            # down_fea = token_fea

            sk_fea = down_fea[:batch // 2]
            im_fea = down_fea[batch // 2:]

            cos_scores = cos_similar(sk_fea, im_fea)
            cos_scores = cos_scores.view(batch // 2, -1)
            rn_scores = self.rn(cos_scores)

            # print('cls_fea:', cls_fea.size())
            # print('rn_scores:', cls_fea.size())

            return cls_fea, rn_scores

        else:

            if only_sa:
                sa_fea, left_tokens, idxs = self.sa(sk)  # b*768*197
                return sa_fea, idxs
            else:
                sk_im = torch.cat((sk, im), dim=0)
                # transformer
                ca_fea = self.ca(sk_im, stage)  # b*seq_len*768

                cls_fea = ca_fea[:, 0]
                token_fea = ca_fea[:, 1:]
                batch = token_fea.size(0)

                token_fea = token_fea.view(batch, 768, 14, 14)
                down_fea = self.conv2d(token_fea)
                down_fea = down_fea.view(batch, 512, 7*7)
                down_fea = down_fea.transpose(1, 2)

                # down_fea = self.linear(token_fea)
                # down_fea = token_fea

                sk_fea = down_fea[:batch // 2]
                im_fea = down_fea[batch // 2:]

                cos_scores = cos_similar(sk_fea, im_fea)
                cos_scores = cos_scores.view(batch // 2, -1)
                rn_scores = self.rn(cos_scores)

                return cls_fea, rn_scores


if __name__ == '__main__':
    args = Option().parse()
    print(str(args))

    # if args.number_gpu == 1:
    sk = torch.rand((4, 224, 224))
    im = torch.rand((4, 224, 224))
    model = Model()
    cls_fea, rn_scores = model(sk, im)