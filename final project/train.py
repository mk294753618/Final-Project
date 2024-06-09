import os
import argparse
import tqdm
import logging

import torch
import torch.optim as optim

from torchsummary import summary

from terminaltables import AsciiTable

from utils.parse_config import parse_data_config
from utils.utils import load_classes, to_cpu
from models import load_model
from utils.data import create_data_loader, create_valid_data_loader
from utils.loss import compute_loss
from utils.logger import Logger
from evaluate import _evaluate


def run():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='YOLOv3 train')
    parser.add_argument('--model', type=str, default='./config/yolov3_spp.cfg', help='path to model file')
    parser.add_argument('--data', type=str, default='./data/valorant.data', help='path to classes name file')
    parser.add_argument('--epoch', type=int, default=30, help='num of epochs')
    parser.add_argument('--verbose', action='store_true', help='show detail of training')
    parser.add_argument('--n_cpu', type=int, default=0, help='num_worker')
    parser.add_argument('--pretrained_path', type=str, help='path to pretrained path file')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='between how many epoch saves the checkpoint')
    parser.add_argument('--evaluation_interval', type=int, default=1,
                        help='between how many epoch evaluate the validation set')
    parser.add_argument('--multiscale_training', action='store_true', help='allow multiscale_training')
    parser.add_argument('--IOU_thres', type=float, default=0.5, help='IOU threshold')
    parser.add_argument('--CON_thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--NMS_thres', type=float, default=0.5, help='IOU threshold')
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")

    args = parser.parse_args()
    logging.info(f"Command line arguments: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    data_config = parse_data_config(args.data)
    train_path = data_config['train']
    valid_path = data_config['valid']
    class_names = load_classes(data_config['names'])

    model = load_model(device, args.model, './yolov3-spp-ultralytics-416.pt')
    logger = Logger(args.logdir)
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    data_loader = create_data_loader(
        path=train_path,
        batch_size=mini_batch_size,
        img_size=model.hyperparams['height'],
        n_cpu=args.n_cpu,
        multiscale_training=args.multiscale_training,
    )
    valid_data_loader = create_valid_data_loader(
        path=valid_path,
        batch_size=mini_batch_size,
        img_size=model.hyperparams['height'],
        n_cpu=args.n_cpu,
    )

    param = [p for p in model.parameters() if p.requires_grad]

    if model.hyperparams['optimizer'] == 'adam':
        optimizer = optim.Adam(
            param,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay']
        )
    elif model.hyperparams['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            param,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum']
        )
    else:
        logging.error('unknown optimizer')
        return

    for epoch in range(1, args.epoch+1):
        logging.info('\n---training model---')
        model.train()

        for i, (_, imgs, targets) in enumerate(tqdm.tqdm(data_loader, desc=f'Training Epoch {epoch}')):
            batch_done = len(data_loader) * epoch + i
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            output = model(imgs)

            loss, loss_component = compute_loss(output, targets, model)

            loss.backward()
            if batch_done % model.hyperparams['subdivisions'] == 0:
                lr = model.hyperparams['learning_rate']
                if batch_done < model.hyperparams['burn_in']:
                    lr *= (batch_done / model.hyperparams['burn_in'])
                else:
                    for threshold, value in model.hyperparams['lr_steps']:
                        if int(batch_done) > int(threshold):
                            lr *= value
                logger.scalar_summary("train/learning_rate", lr, batch_done)
                for g in optimizer.param_groups:
                    g['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

            if args.verbose:
                logging.info(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_component[0])],
                        ["Object loss", float(loss_component[1])],
                        ["Class loss", float(loss_component[2])],
                        ["Loss", float(loss_component[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            tensorboard_log = [
                ("train/iou_loss", float(loss_component[0])),
                ("train/obj_loss", float(loss_component[1])),
                ("train/class_loss", float(loss_component[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batch_done)

            model.seen += imgs.size(0)

        if epoch % args.checkpoint_interval == 0 and epoch > 20:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            logging.info(f"--- Saving checkpoint to: '{checkpoint_path}' ---")
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % args.evaluation_interval == 0:
            logging.info("\n--- evaluating model ---")

            metrics_output = _evaluate(
                model,
                valid_data_loader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.IOU_thres,
                conf_thres=args.CON_thres,
                nms_thres=args.NMS_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)




if __name__ == "__main__":
    run()


