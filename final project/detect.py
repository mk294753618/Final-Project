import os
import random
import tqdm
import argparse

import torch
from torch.autograd import Variable

from models import load_model
from utils.data import create_data_loader_detect
from utils.utils import nms, rescale_boxes, load_classes



import cv2


def detect(model, dataloader, path, conf_thres, nms_thres):
    os.makedirs(path, exist_ok=True)

    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []
    imgs = []

    for (img_path, img) in tqdm.tqdm(dataloader, desc='Detecting'):
        img = Variable(img.type(Tensor))

        with torch.no_grad():
            detections = model(img)
            detections = nms(detections, conf_thres, nms_thres)

        img_detections.extend(detections)
        imgs.extend(img_path)

    return img_detections, imgs


def draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    for (image_path, detection) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        draw_and_save_output_image(
            image_path, detection, img_size, output_path, classes)


def draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    detections = rescale_boxes(detections, img_size, img.shape[:2])

    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        if int(cls_pred) == 0:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        img = cv2.putText(img, '{} {:.3f}'.format(classes[int(cls_pred)], conf), (int(x1), int(y1)),
                          font, 1, (int(x2), int(y2)), 4)
        filename = os.path.basename(image_path).split(".")[0]
        output_path = os.path.join(output_path, f"{filename}.png")
        cv2.imwrite(output_path, img)


def detect_directory(model_path, weights_path, img_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    dataloader = create_data_loader_detect(img_path, batch_size, img_size, n_cpu)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device, model_path, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        conf_thres,
        nms_thres)
    draw_and_save_output_images(
        img_detections, imgs, img_size, output_path, classes)

    print(f"---- Detections were saved to: '{output_path}' ----")

def run():
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument('--model', type=str, default='./config/yolov3_spp.cfg', help='path to model file')
    parser.add_argument("--weights", type=str, default="./checkpoints/yolov3_ckpt_30.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("--images", type=str, default="./samples", help="Path to directory with images to inference")
    parser.add_argument("--classes", type=str, default="./data/valorant.names", help="Path to classes label file (.names)")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=0, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)


if __name__ == '__main__':
    run()









