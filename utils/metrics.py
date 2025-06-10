import torch
from .comm import save_binary_img, save_gt_img
from opt import opt
import numpy as np


def eval_smeasure(pred, gt, alpha=0.5):
    """
    计算 S-measure
    """
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        Q = alpha * s_object(pred, gt) + (1 - alpha) * s_region(pred, gt)
        if Q.item() < 0:
            Q = torch.tensor(0.0, device=pred.device)
    return Q.item()


def s_object(pred, gt):
    """
    计算 S-measure 中的 object 部分
    """
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def _object(pred, gt):
    """
    计算 object 分数
    """
    temp = pred[gt == 1]
    if temp.numel() == 0:  # 防止除以0错误
        return torch.tensor(0.0, device=pred.device)
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    return score


def s_region(pred, gt):
    """
    计算 S-measure 中的 region 部分
    """
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1)
    Q2 = ssim(p2, gt2)
    Q3 = ssim(p3, gt3)
    Q4 = ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


def centroid(gt):
    """
    计算质心
    """
    device = gt.device
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)

    if gt.sum() == 0:
        X = torch.tensor(round(cols / 2), device=device)
        Y = torch.tensor(round(rows / 2), device=device)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float().to(device)
        j = torch.from_numpy(np.arange(0, rows)).float().to(device)
        X = torch.round((gt.sum(dim=0) * i).sum() / total).long()
        Y = torch.round((gt.sum(dim=1) * j).sum() / total).long()

    return X, Y


def divideGT(gt, X, Y):
    """
    划分 GT
    """
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]

    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def dividePrediction(pred, X, Y):
    """
    划分预测
    """
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def ssim(pred, gt):
    """
    计算 SSIM
    """
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x).pow(2)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y).pow(2)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)
    alpha = 4 * x * y * sigma_xy
    beta = (x.pow(2) + y.pow(2)) * (sigma_x2 + sigma_y2)
    Q = alpha / (beta + 1e-20) if beta != 0 else 1.0
    return Q


def eval_emeasure(y_pred, y, num=255, cuda=True):
    if cuda:
        score = torch.zeros(num).cuda()
    else:
        score = torch.zeros(num)

    for i in range(num):
        fm = y_pred - y_pred.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score.max()


def evaluate(pred, gt, name=None, testset=None):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    # pred = torch.sigmoid(pred)
    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    if opt.save_img == True  and opt.load_ckpt is not None:
        save_binary_img(pred_binary, testset, name)
        save_gt_img(gt_binary, testset, name)

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F-measure
    Fmeasure = (1.3) * Precision * Recall / ((0.3) * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    mIou = TP / (TP + FP + FN)

    # mae
    mae = torch.abs(pred - gt).mean().item()

    # Smeasure
    Smeasure = eval_smeasure(pred, gt)

    # Emeasure
    Emeasure = eval_emeasure(pred, gt)

    # IoU for background
    # IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    # IoU_mean = (mIou + IoU_bg) / 2.0

    return Recall, Specificity, Precision, F1, Fmeasure, ACC_overall, mIou, mae, Smeasure, Emeasure


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v


    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics
