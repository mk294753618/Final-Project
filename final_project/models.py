from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from utils.parse_config import parse_model_config



def create_model(module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]:
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'decay': float(hyperparams['decay']),
        'momentum': float(hyperparams['momentum']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int, hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams['width'] == hyperparams['height'], \
        "Height and width should be equal"

    output_filters = [hyperparams['channels']]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        module = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2

            module.add_module(f'Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                    out_channels=filters,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=pad,
                                                    bias=not bn))
            if bn:
                module.add_module(f'BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def['activation'] == 'leaky':
                module.add_module(f'leaky_relu_{i}', nn.LeakyReLU(0.1))

        elif module_def["type"] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                module.add_module(f'_debug_padding', nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            module.add_module(f'maxpool', maxpool)

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            module.add_module(f'shortcut', nn.Sequential())

        elif module_def['type'] == 'upsample':
            stride = int(module_def['stride'])
            module.add_module(f'upsample', Upsample(scale_factor=stride))

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][x] for x in layers])
            module.add_module(f'route', nn.Sequential())

        elif module_def['type'] == 'yolo':
            anchor_index = [int(x) for x in module_def['mask'].split(',')]
            anchor = [int(x) for x in module_def['anchors'].split(',')]
            anchor = [(anchor[i], anchor[i+1]) for i in range(0, len(anchor), 2)]
            anchor = [anchor[i] for i in anchor_index]
            num_classes = int(module_def['classes'])
            module.add_module(f'yolo_{i}', YOLOLayer(anchor, num_classes))

        module_list.append(module)
        output_filters.append(filters)

    return hyperparams, module_list



class Upsample(nn.Module):

    def __init__(self, scale_factor: int, mode: str = 'nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = func.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module):

    def __init__(self, anchor: List[Tuple[int, int]], num_classes: int):
        super(YOLOLayer, self).__init__()
        self.na = len(anchor)
        self.no = num_classes + 5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.grid = torch.zeros(1)

        anchor = torch.tensor(list(chain(*anchor))).float().view(-1,2)
        self.register_buffer('anchor', anchor)
        self.register_buffer('anchor_grid', anchor.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)

        return x
    @staticmethod
    def _make_grid(nx: int, ny: int):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2)


class Darknet(nn.Module):

    def __init__(self, path, init_weight=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(path)
        self.hyperparams, self.module_list = create_model(self.module_defs)
        self.yolo_layer = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, self.seen, 0], dtype=np.int32)
        if init_weight:
            self._initialize_weight()

    def _initialize_weight(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.normal_(i.weight.data, 0.0, 0.02)
            elif isinstance(i, nn.BatchNorm2d):
                nn.init.normal_(i.weight.data, 1.0, 0.02)
                nn.init.constant_(i.bias.data, 0.0)

    def forward(self, x):
        img_size = x.size(2)
        output = []
        yolo_output = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                combined_outputs = torch.cat(
                [output[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id: group_size * (group_id + 1)]
            elif module_def['type'] == 'shortcut':
                layer = int(module_def['from'])
                x = output[-1] + output[layer]
            elif module_def['type'] == 'yolo':
                x = module[0](x, img_size)
                yolo_output.append(x)
            output.append(x)
        return yolo_output if self.training else torch.cat(yolo_output, 1)

    def save_darknet_weight(self, path):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.model_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
            fp.close()

    def load_darknet_weights(self, path):
        with open(path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weight = np.fromfile(f, dtype=np.float)

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.model_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                bn_layer = module[1]
                num_bias = bn_layer.bias.numel()
                bn_b = torch.from_numpy(weight[ptr:ptr+num_bias]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_bias
                bn_w = torch.from_numpy(weight[ptr:ptr+num_bias]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_bias
                bn_rm = torch.from_numpy(weight[ptr:ptr+num_bias]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_bias
                bn_rv = torch.from_numpy(weight[ptr:ptr+num_bias]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_bias
                num_weight = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weight[ptr:ptr+num_weight]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_weight


def load_model(device, model_path: str, weight_path: str = None):
    model = Darknet(model_path, True).to(device)

    if weight_path:
        if weight_path.endswith('.pth'):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        elif weight_path.endswith('.pt'):
            weight = weight_path.format(416)
            weight =torch.load(weight, map_location=device)['model']
            del weight['module_list.88.Conv2d.weight']
            del weight['module_list.88.Conv2d.bias']
            del weight['module_list.100.Conv2d.weight']
            del weight['module_list.100.Conv2d.bias']
            del weight['module_list.112.Conv2d.weight']
            del weight['module_list.112.Conv2d.bias']
            model.load_state_dict(weight, strict=False)
        else:
            model.load_darknet_weights(weight_path)

    return model



