import numpy as np
import torch
import random
import torchvision
import tqdm


def load_classes(path):
    with open(path, 'r') as fp:
        names = fp.read().splitlines()
    return names


def to_cpu(tensor):
    return tensor.detach().cpu()

def worker_seed_set(worker_id):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, pred_cls, conf = tp[i], pred_cls[i], conf[i]

    unique_classes = np.unique(target_cls)
    ap, precision, recall = [], [], []

    for c in tqdm.tqdm(unique_classes, desc='computing AP'):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            precision.append(0)
            recall.append(0)
        else:
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            recall_curve = tpc / (n_gt + 1e-16)
            precision_curve = tpc / (tpc + fpc)

            recall.append(recall_curve[-1])
            precision.append(precision_curve[-1])

            ap.append(compute_ap(recall_curve, precision_curve))

    precision = np.array(precision)
    recall = np.array(recall)
    ap = np.array(ap)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, ap, f1, unique_classes.astype('int32')



def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for i in range(len(outputs)):
        if outputs[i] is None:
            continue

        output = outputs[i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        tp = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detect_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if len(detect_boxes) == len(annotations):
                    break

                if pred_label not in target_labels:
                    continue

                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))
                iou, box_filter_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)
                box_index = filtered_target_position[box_filter_index]

                if iou >= iou_threshold and box_index not in detect_boxes:
                    tp[pred_i] = 1
                    detect_boxes += [box_index]
        batch_metrics.append([tp, pred_scores, pred_labels])
    return batch_metrics


def bbox_iou(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou



def nms(prediction, conf_thres=0.25, iou_thres=0.45):
    max_wh = 4096
    max_det = 5
    max_nms = 500

    output = [torch.zeros((0, 6), device='cpu')] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[x[..., 4] > conf_thres]

        if not x.shape[0]:
            continue

        box = x.clone()
        x[:, 5:] *= x[:, 4:5]
        box[:, 0] = x[:, 0] - x[:, 2] / 2
        box[:, 1] = x[:, 1] - x[:, 3] / 2
        box[:, 2] = x[:, 0] + x[:, 2] / 2
        box[:, 3] = x[:, 1] + x[:, 3] / 2
        box = box[:, :4]

        i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        x = torch.cat((box[i], x[i, j+5, None], j[:, None].float()), 1)
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes = x[:, :4] + c
        score = x[:, 4]
        i = torchvision.ops.nms(boxes, score, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

    return output


def rescale_boxes(boxes, current_dim, original_shape):
    orig_h, orig_w = original_shape

    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes





