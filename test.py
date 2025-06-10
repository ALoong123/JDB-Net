
import os
from tqdm import tqdm
from opt import opt

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


import torch
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
from utils.comm import generate_model

from utils.metrics import Metrics
from utils.metrics import evaluate


def test(model, test_data_dir):
    test_data_name = test_data_dir.split("/")[1]

    print('Loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)

    model.eval()

    # metrics_logger initialization  mae, f_measure, e_measure, s_measure
    metrics = Metrics(
        ['recall', 'specificity', 'precision', 'F1', 'Fmeasure', 'ACC_overall', 'mIou', 'mae', 'Smeasure', 'Emeasure'])

    print('Start testing')
    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:
            img, gt, name = data['image'], data['label'], data['name']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _Fmeasure, _ACC_overall, _mIou, _mae, _Smeasure, _Emeasure = evaluate(
                output, gt, name,
                test_data_name)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision, F1=_F1, Fmeasure=_Fmeasure,
                           ACC_overall=_ACC_overall, mIou=_mIou, mae=_mae, Smeasure=_Smeasure, Emeasure=_Emeasure)

    metrics_result = metrics.mean(total_batch)

    results = open('./checkpoints/' + str(opt.expID) + "/testResults.txt", "a+")

    print("\n%s Test Result:" % test_data_name, file=results)
    print(
        'recall: %.3f, specificity: %.3f, precision: %.3f, F1: %.3f, Fmeasure: %.3f, ACC_overall: %.3f, mIou: %.3f,mae: %.3f, Smeasure: %.3f, Emeasure: %.3f' % (
            metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'],
            metrics_result['Fmeasure'], metrics_result['ACC_overall'], metrics_result['mIou'], metrics_result['mae'],
            metrics_result['Smeasure'], metrics_result['Emeasure']), file=results)

    print("\n%s Test Result:" % test_data_name)
    print(
        'recall: %.3f, specificity: %.3f, precision: %.3f, F1: %.3f, Fmeasure: %.3f, ACC_overall: %.3f, mIou: %.3f,,mae: %.3f, Smeasure: %.3f, Emeasure: %.3f' % (
            metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'],
            metrics_result['Fmeasure'], metrics_result['ACC_overall'], metrics_result['mIou'], metrics_result['mae'],
            metrics_result['Smeasure'], metrics_result['Emeasure']))

    results.close()


if __name__ == '__main__':

    print('Loading model......')
    model = generate_model(opt)

    test_data_list = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]


    for test_data_dir in test_data_list:
        test(model, "test/" + test_data_dir)



    print('Test Done')
