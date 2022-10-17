import torch
import torch.nn as nn


def triplet_loss(x, args):
    """

    :param x: batch*4->sk_p,sk_n,im_p,im_n
    :param args:
    :return:
    """
    triplet = nn.TripletMarginLoss(margin=args.margin, p=args.p).cuda()
    # triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    sk_p = x[0:args.batch]
    im_p = x[2 * args.batch:3 * args.batch]
    im_n = x[3 * args.batch:]
    loss = triplet(sk_p, im_p, im_n)

    return loss

def triplet_loss2(a,p,n, args):
    """

    :param x: batch*4->sk_p,sk_n,im_p,im_n
    :param args:
    :return:
    """
    triplet = nn.TripletMarginLoss(margin=args.margin, p=args.p)
    loss = triplet(a, p, n)

    return loss



def rn_loss(predict, target):
    # mse_loss = nn.functional.mse_loss()
    mse_loss = nn.MSELoss().cuda()
    loss = mse_loss(predict, target)

    return loss


def h_loss(predict, target, neg):

    # .. math::
    #     l_n = \begin{cases}
    #         x_n, & \text{if}\; y_n = 1,\\
    #         \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
    #     \end{cases}

    h_loss = nn.HingeEmbeddingLoss(margin=-0.01).cuda()

    input = -((predict-target)**2)

    # zeros = torch.cat((torch.zeros(10), torch.zeros(10)), dim=0).cuda()
    # loss = torch.mean(torch.max(input=zeros, other=-0.01-input))*6
    # print(loss)

    loss = h_loss(input, neg)

    return loss

def cos_loss(xrn1, xrn2, target):

    # .. math::
    #     \text{loss}(x, y) =
    #     \begin{cases}
    #     1 - \cos(x_1, x_2), & \text{if } y = 1 \\
    #     \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
    #     \end{cases}

    cos_loss = nn.CosineEmbeddingLoss(margin=0.).cuda()

    loss = cos_loss(xrn1, xrn2, target)

    return loss


def ece_loss(predict, target):

    bin_boundaries = torch.linspace(0, 1, 10 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    # print("bin_boundaries:", bin_boundaries)

    confidences = predict.squeeze(dim=1)
    accuracies = target.squeeze(dim=1)
    # print(confidences.size(), accuracies.size())
    print("confidences:", confidences)
    # print("accuracies", accuracies)

    ece = torch.zeros(1).cuda()
    m = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        m += 1
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        # print(in_bin)     # [True, False]
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # print(accuracies[in_bin])
            # print("bin", m, ": accu_mean:", accuracy_in_bin)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # print(confidences[in_bin])
            # print("bin", m, ": conf_mean:", avg_confidence_in_bin)
            print("bin", m, ": num:", in_bin.float().sum(),
                  "accu_mean:", accuracy_in_bin,
                  "conf_mean:", avg_confidence_in_bin)
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # print("ece:", ece)

    return ece


def ece_low_loss(predict, target):

    bin_boundaries = torch.linspace(0, 1, 10 + 1)
    bin_lowers = torch.Tensor([0.0, 0.1, 0.8, 0.9])
    bin_uppers = torch.Tensor([0.1, 0.2, 0.9, 1.0])
    # print("bin_boundaries:", bin_boundaries)

    confidences = predict.squeeze(dim=1)
    accuracies = target.squeeze(dim=1)
    # print(confidences.size(), accuracies.size())
    print("confidences:", confidences)
    # print("accuracies", accuracies)

    ece = torch.zeros(1).cuda()
    m = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        m += 1
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        # print(in_bin)     # [True, False]
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # print(accuracies[in_bin])
            # print("bin", m, ": accu_mean:", accuracy_in_bin)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # print(confidences[in_bin])
            # print("bin", m, ": conf_mean:", avg_confidence_in_bin)
            print("bin", m, ": num:", in_bin.float().sum(),
                  "accu_mean:", accuracy_in_bin,
                  "conf_mean:", avg_confidence_in_bin)
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # print("ece:", ece)

    return ece


# class _ECELoss(nn.Module):
#     """
#     Calculates the Expected Calibration Error of a model.
#     (This isn't necessary for temperature scaling, just a cool metric).
#     The input to this loss is the logits of a model, NOT the softmax scores.
#     This divides the confidence outputs into equally-sized interval bins.
#     In each bin, we compute the confidence gap:
#     bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
#     We then return a weighted average of the gaps, based on the number
#     of samples in each bin
#     See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
#     "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
#     2015.
#     """
#     def __init__(self, n_bins=15):
#         """
#         n_bins (int): number of confidence interval bins
#         """
#         super(_ECELoss, self).__init__()
#         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#         self.bin_lowers = bin_boundaries[:-1]
#         self.bin_uppers = bin_boundaries[1:]
#
#     def forward(self, logits, labels):
#         softmaxes = F.softmax(logits, dim=1)
#         confidences, predictions = torch.max(softmaxes, 1)
#         accuracies = predictions.eq(labels)
#
#         ece = torch.zeros(1, device=logits.device)
#         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
#             # Calculated |confidence - accuracy| in each bin
#             in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
#             prop_in_bin = in_bin.float().mean()
#             if prop_in_bin.item() > 0:
#                 accuracy_in_bin = accuracies[in_bin].float().mean()
#                 avg_confidence_in_bin = confidences[in_bin].mean()
#                 ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
#
#         return ece


class TripletHardLoss:

    def __init__(self, margin=1):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, inputs, targets, index=None):
        """
        inputs: batch * dim
        target batch
        index: 只针对sketch 找寻image
        """
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(mat1=inputs, mat2=inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if index is None:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        else:
            # print(n,index)
            # 限定前index是sketch
            for i in range(n):
                if i < n / 2:
                    # sketch 找 image
                    dist_ap.append(((dist[i][mask[i]])[index:]).max().unsqueeze(0))
                    dist_an.append(((dist[i][mask[i] == 0])[index:]).min().unsqueeze(0))
                else:
                    # image 找 sketch
                    dist_ap.append(((dist[i][mask[i]])[:index]).max().unsqueeze(0))
                    dist_an.append(((dist[i][mask[i] == 0])[:index]).min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
