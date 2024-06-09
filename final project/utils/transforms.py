import torch
import torch.nn.functional as func
from torchvision.transforms import transforms
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage


class ImgAug(object):

    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        img, boxes = data
        boxes = np.array(boxes)
        boxes_cpy = boxes.copy()
        boxes[:, 1] = boxes_cpy[:, 1] - boxes_cpy[:, 3] / 2
        boxes[:, 2] = boxes_cpy[:, 2] - boxes_cpy[:, 4] / 2
        boxes[:, 3] = boxes_cpy[:, 1] + boxes_cpy[:, 3] / 2
        boxes[:, 4] = boxes_cpy[:, 2] + boxes_cpy[:, 4] / 2

        bboxes = BoundingBoxesOnImage([BoundingBox(*box[1:], label=box[0]) for box in boxes],
                                      shape=img.shape)

        img, bboxes = self.augmentations(image=img, bounding_boxes=bboxes)

        bboxes = bboxes.clip_out_of_image()
        boxes = np.zeros((len(bboxes), 5))

        for i, box in enumerate(bboxes):
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            boxes[i, 0] = box.label
            boxes[i, 1] = (x1 + x2) / 2
            boxes[i, 2] = (y1 + y2) / 2
            boxes[i, 3] = x2 - x1
            boxes[i, 4] = y2 - y1

        return img, boxes


class RelativeLabels(object):

    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self):
        self.augmentations = iaa.Sequential(
            [iaa.PadToAspectRatio(1.0, position='center-center').to_deterministic()]
        )


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        img = transforms.ToTensor()(img)

        bboxes = torch.zeros((len(boxes), 6))
        bboxes[:, 1:] = transforms.ToTensor()(boxes)
        return img, bboxes


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = func.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


class DefaultAug(ImgAug):
    def __init__(self):
        self.augmentations = iaa.Sequential(
            [iaa.Sharpen((0.0, 0.1)),
             iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
             iaa.AddToBrightness((-60, 40)),
             iaa.AddToHue((-10, 10)),
             iaa.Fliplr(0.5),
             ]
        )


class StrongAug(ImgAug):
    def __init__(self):
        self.augmentations = iaa.Sequential(
            [iaa.Dropout([0.0, 0.01]),
             iaa.Sharpen((0.0, 0.1)),
             iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
             iaa.AddToBrightness((-60, 40)),
             iaa.AddToHue((-20, 20)),
             iaa.Fliplr(0.5),
             ]
        )

DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

STRONG_AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    StrongAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])



