import torch
from torch.autograd import Variable
import numpy as np
import logging

import tqdm

from terminaltables import AsciiTable

from models import load_model
from utils.utils import load_classes, ap_per_class, nms, get_batch_statistics
from utils.parse_config import parse_data_config
from utils.data import create_valid_data_loader

def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = []
    sample_metrics = []
    for _, imgs, targets in tqdm.tqdm(dataloader, desc='Validation'):
        labels += targets[:, 1].tolist()
        targets_cpy = targets.clone()
        targets[:, 2] = targets_cpy[:, 2] - targets_cpy[:, 4] / 2
        targets[:, 3] = targets_cpy[:, 3] - targets_cpy[:, 5] / 2
        targets[:, 4] = targets_cpy[:, 2] + targets_cpy[:, 4] / 2
        targets[:, 5] = targets_cpy[:, 3] + targets_cpy[:, 5] / 2
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            output = model(imgs)
            output = nms(output, conf_thres, nms_thres)

        sample_metrics += get_batch_statistics(output, targets, iou_thres)

    if len(sample_metrics) == 0:
        tqdm.tqdm.write('---No detections---')
        return None

    tp, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]
    metrics_output = ap_per_class(tp, pred_scores, pred_labels, labels)
    print_eval_stats(metrics_output, class_names, verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, ap, f1, classes = metrics_output
        if verbose:
            ap_table = [['index', 'class', 'ap']]
            for i, c in enumerate(classes):
                ap_table += [[c, class_names[c], '%.5f' % ap[i]]]
            tqdm.tqdm.write(AsciiTable(ap_table).table)
        logging.info(f'\n---map:{ap.mean():.5f}---')
    else:
        logging.error('---map not measure---')


def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=0, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    dataloader = create_valid_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose)
    return metrics_output

