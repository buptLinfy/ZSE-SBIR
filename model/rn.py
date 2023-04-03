import torch
import torch.nn as nn
from torch import Tensor


class Relation_Network(nn.Module):
    def __init__(self, anchor, dropout=0.1):
        super(Relation_Network, self).__init__()

        # 1.0
        if anchor == 49:
            self.rn = nn.Sequential(
                nn.Linear(anchor * anchor, 343, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(343, 49, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(49, 1, bias=True)
            )

        # 1.0
        elif anchor == 196:
            self.rn = nn.Sequential(
                nn.Linear(anchor * anchor, 2744, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(2744, 196, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(196, 1, bias=True)
            )

        # 0.9
        elif anchor == 144:
            self.rn = nn.Sequential(
                nn.Linear(anchor * anchor, 1728, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(1728, 144, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(144, 1, bias=True)
            )

        else:
            raise Exception

    def forward(self, x):
        """
        :param x: sketch、image concat-->b*
        :return:
        """
        x = self.rn(x)
        x = torch.sigmoid(x)
        return x


class Scale_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        return self.seq(x)


def cos_similar(p: Tensor, q: Tensor):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix
