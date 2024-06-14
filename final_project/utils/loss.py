import torch
import torch.nn as nn

from .utils import to_cpu

def build_target(p, targets, model):
    na = 3
    nt = targets.shape[0]
    device = targets.device
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=device)
    ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    for i, yolo_layer in enumerate(model.yolo_layer):
        anchor = yolo_layer.anchor / yolo_layer.stride
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        t = targets * gain

        if nt:
            r = t[:, :, 4:6] / anchor[:, None]
            j = torch.max(r, 1/r).max(2)[0] < 4
            t = t[j]
        else:
            t = targets[0]

        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = gxy.long()
        gi, gj = gij.T
        a = t[:, 6].long()

        indices.append((b, a, gj.clamp_(0, gain[3].long()-1), gi.clamp_(0, gain[4].long()-1)))
        tbox.append(torch.cat((gxy-gij, gwh), 1))
        anch.append(anchor[a])
        tcls.append(c)

    return tcls, tbox, indices, anch


def d_iou(box_1, box_2):
    box_2 = box_2.T
    b1_x1 = box_1[0] - box_1[2] / 2
    b1_x2 = box_1[0] + box_1[2] / 2
    b1_y1 = box_1[1] - box_1[3] / 2
    b1_y2 = box_1[1] + box_1[3] / 2
    b2_x1 = box_2[0] - box_2[2] / 2
    b2_x2 = box_2[0] + box_2[2] / 2
    b2_y1 = box_2[1] - box_2[3] / 2
    b2_y2 = box_2[1] + box_2[3] / 2


    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = box_1[2] * box_1[3] + box_2[2] * box_2[3] - inter
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    c2 = cw**2 + ch**2
    r = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    return iou - r / c2


def compute_loss(p, targets, model):

    device = targets.device
    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)
    tcls, tbox, indices, anch = build_target(p, targets, model)

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    for i, layer_predict in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(layer_predict[..., 0], device=device)
        num_target = b.size(0)

        if num_target:
            ps = layer_predict[b, a, gj, gi]
            pxy = ps[:, :2].sigmoid()
            pwh = torch.exp(ps[:, 2:4]) * anch[i]
            pbox = torch.cat((pxy, pwh), 1)
            diou = d_iou(pbox.T, tbox[i])
            lbox += (1.0 - diou).mean()

            tobj[b, a, gj, gi] = diou.detach().clamp(0).type(tobj.dtype)

            if ps.size(1) - 5 > 1:
                t = torch.zeros_like(ps[:, 5:], device=device)
                t[range(num_target), tcls[i]] = 1
                lcls += BCEcls(ps[:, 5:], t)

        lobj += BCEobj(layer_predict[..., 4], tobj)

    loss = 0.05 * lbox + 1.0 * lobj + 0.5 * lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))










