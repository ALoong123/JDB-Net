from opt import opt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics_vaild import evaluate, Metrics
from utils.comm import generate_model
from utils.loss import DeepSupervisionLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard


def valid(model, valid_dataloader, total_batch):
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(
        ['recall', 'specificity', 'precision', 'F1', 'Fmeasure', 'ACC_overall', 'mIou'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)

            _recall, _specificity, _precision, _F1, _Fmeasure, _ACC_overall, _mIou = evaluate(output, gt)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision, F1=_F1, Fmeasure=_Fmeasure,
                           ACC_overall=_ACC_overall, mIou=_mIou)

    metrics_result = metrics.mean(total_batch)

    return metrics_result


def train():
    # load model
    print('Loading model......')
    model = generate_model(opt)
    print('Load model:', opt.model)

    # load data
    print('Loading data......')
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train')
    train_dataloader = DataLoader(train_data, int(opt.batch_size), shuffle=True, num_workers=opt.num_workers)
    valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: pow(1.0 - epoch / opt.nEpoch, opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')

    os.makedirs('./checkpoints/' + str(opt.expID), exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=f'./checkpoints/{opt.expID}/logs')

    results = open('./checkpoints/' + str(opt.expID) + "/validResults.txt", "a+")
    best_IoU = 0
    best_F1 = 0
    best_idx1 = 0
    best_idx2 = 0

    size_rates = [0.75, 1, 1.25]

    for epoch in range(opt.nEpoch):
        print('------ Epoch', epoch + 1 + 0)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)
        bar = tqdm(enumerate(train_dataloader), total=total_batch)

        lossA = 0
        for i, data in bar:
            img = data['image']
            gt = data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            for rate in size_rates:

                trainsize = int(round(352 * rate / 32) * 32)
                if rate != 1:
                    img = F.interpolate(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gt = F.interpolate(gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                optimizer.zero_grad()
                output = model(img)
                loss = DeepSupervisionLoss(output, gt)
                lossA += loss
                loss.backward()

                optimizer.step()
                bar.set_postfix_str('loss: %.5s' % loss.item())

                # 记录训练损失到 TensorBoard
                # writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/Loss', lossA.item() / (3 * 81), epoch)

        scheduler.step()

        metrics_result = valid(model, valid_dataloader, val_total_batch)

        # 记录验证集的评价指标到 TensorBoard
        writer.add_scalar('Valid/Recall', metrics_result['recall'], epoch)
        writer.add_scalar('Valid/Specificity', metrics_result['specificity'], epoch)
        writer.add_scalar('Valid/Precision', metrics_result['precision'], epoch)
        writer.add_scalar('Valid/F1', metrics_result['F1'], epoch)
        writer.add_scalar('Valid/Fmeasure', metrics_result['Fmeasure'], epoch)
        writer.add_scalar('Valid/ACC_overall', metrics_result['ACC_overall'], epoch)
        writer.add_scalar('Valid/mIou', metrics_result['mIou'], epoch)

        print("\nValid Result of epoch %d:" % (epoch + 1 + 0), file=results)
        print(
            'recall: %.3f, specificity: %.3f, precision: %.3f, F1: %.3f, Fmeasure: %.3f, ACC_overall: %.3f, mIou: %.3f' % (
                metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                metrics_result['F1'],
                metrics_result['Fmeasure'], metrics_result['ACC_overall'], metrics_result['mIou']), file=results)
        print("\nValid Result of epoch %d:" % (epoch + 1 + 0))
        print(
            'recall: %.3f, specificity: %.3f, precision: %.3f, F1: %.3f, Fmeasure: %.3f, ACC_overall: %.3f, mIou: %.3f' % (
                metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                metrics_result['F1'],
                metrics_result['Fmeasure'], metrics_result['ACC_overall'], metrics_result['mIou']))

        if ((epoch + 1 + 0) % opt.ckpt_period == 0):
            torch.save(model.state_dict(), './checkpoints/' + str(opt.expID) + "/ck_{}.pth".format(epoch + 1 + 0))

        if metrics_result['mIou'] > best_IoU:
            best_idx1 = epoch + 1 + 0
            best_IoU = metrics_result['mIou']
            torch.save(model.state_dict(),
                       './checkpoints/' + str(opt.expID) + "/bestIou_ck_{}.pth".format(epoch + 1 + 0))

        if metrics_result['F1'] > best_F1:
            best_idx2 = epoch + 1 + 0
            best_F1 = metrics_result['F1']
            torch.save(model.state_dict(),
                       './checkpoints/' + str(opt.expID) + "/bestF1_ck_{}.pth".format(epoch + 1 + 0))
        print("Epoch %d with best mIoU: %.3f,Epoch %d with best mF1:%.3f" % (best_idx1, best_IoU, best_idx2, best_F1))
    print("\nEpoch %d with best mIoU: %.3f,Epoch %d with best mF1:%.3f" % (best_idx1, best_IoU, best_idx2, best_F1),
          file=results)

    results.close()
    writer.close()  # 关闭 TensorBoard writer


if __name__ == '__main__':
    if opt.mode == 'train':
        print('---PolypSeg Train---')
        train()

    print('Done')
