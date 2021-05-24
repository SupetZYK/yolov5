import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import math
import numpy as np
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

import torch.utils.model_zoo as model_zoo
import math
from box import Anchors, calc_iou
from utils.loss import smooth_l1_loss
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

__all__ = ['ResnetFPN']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1) # bs h w c
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.nc = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        # self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        # out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.nc)
        return out2.contiguous().view(x.shape[0], -1, self.nc)


class PyramidFeatures(nn.Module):
    """ Retinanet Style
    """
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
        # return [P3_x, P4_x, P5_x]




class Model(nn.Module):
    def __init__(self, cfg='retinanet.yaml', nc=None):
        """
        NOTE: this interface is experimental.
        """
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.nc = self.yaml['nc']
        fpn_fs = self.yaml.get('fpn_feat_size', 256)
        depth = self.yaml.get('depth', 50)
        pretrained = self.yaml.get('pretrain', True)
        # if anchors:
        #     logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
        #     self.yaml['anchors'] = round(anchors)  # override yaml value

        super(Model, self).__init__()
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
        elif depth  == 101:
            layers = [3, 4, 13, 3]
            block = Bottleneck
        else:
            raise NotImplementedError()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_fs)
        self.anchor_gen = Anchors()
        self.regressionModel = RegressionModel(fpn_fs)
        self.classificationModel = ClassificationModel(fpn_fs, num_classes=self.nc)
        stride = torch.Tensor(self.anchor_gen.strides)
        self.register_buffer('stride', stride)

        self.names = [str(i) for i in range(self.nc)]  # default names
        self.inplace = self.yaml.get('inplace', True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for modules in [self.classificationModel, self.regressionModel]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        prior = 0.01
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        if pretrained:
            state = model_zoo.load_url(model_urls['resnet{}'.format(depth)])
            res = self.load_state_dict(state, strict=False)
            del state
            print(res)

    @property
    def device(self):
        return self.stride.device

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def forward(self, imgs, **kwargs):
        # shape
        _, _, H, W = imgs.shape
        x = self.conv1(imgs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        self.anchors = self.anchor_gen._get_anchor_boxes(input_size=(W, H), device=self.device, anchor_format='xyxy')
        loc_preds = torch.cat([self.regressionModel(f) for f in features], dim=1) # bs na 4
        cls_preds = torch.cat([self.classificationModel(f) for f in features], dim=1) # bs na nc
        
        # transform
        anchors_xywh = Anchors.xyxy2xywh(self.anchors)
        pred_boxes = self.anchor_gen.predict_transform(loc_preds, anchors_xywh)
        # import ipdb;ipdb.set_trace()
        pred_boxes = self.anchor_gen.xyxy2xywh(pred_boxes)

        pred = torch.cat([pred_boxes, torch.sigmoid(cls_preds)], -1)

        return pred, cls_preds, loc_preds

    def loss(self, classifications, regressions, anchors, targets, **kargs):
        alpha = 0.25
        gamma = 2.0
        num_images = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchors = anchors.view(-1, 4)
        anchor_widths  = (anchors[:, 2] - anchors[:, 0]).unsqueeze(0)
        anchor_heights = (anchors[:, 3] - anchors[:, 1]).unsqueeze(0)
        anchor_ctr_x   = 0.5 * (anchors[:, 0] + anchors[:, 2]).unsqueeze(0)
        anchor_ctr_y   = 0.5 * (anchors[:, 1] + anchors[:, 3]).unsqueeze(0)

        # match anno to anchors first
        with torch.no_grad():
            gt_labels = []
            gt_boxes = []
            for j in range(num_images):
                target = targets[targets[:,0]==j]
                bbox_annotation = target[:,2:]
                bbox_annotation = self.anchor_gen.xywh2xyxy(bbox_annotation)
                anno_label = target[:,1]
                if bbox_annotation.shape[0] == 0:
                    gt_labels_j = torch.zeros(classifications.shape[1]).to(self.device) + self.nc # num_anchors
                    gt_boxes_j = torch.zeros_like(anchors) # num_anchors, 4
                else:
                    IoU = calc_iou(anchors, bbox_annotation) # num_anchors x num_annotations
                    IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                    
                    gt_labels_j = anno_label[IoU_argmax] # num_anchors
                    gt_boxes_j = bbox_annotation[IoU_argmax] # num_anchors, 4

                    # assign bg, ignore, pos based on IoU_max
                    # pos_mask = IoU_max >= 0.5
                    neg_mask = IoU_max <= 0.4
                    ignore_mask = (IoU_max > 0.4) & (IoU_max < 0.5)

                    gt_labels_j[neg_mask] = self.nc
                    gt_labels_j[ignore_mask] = -1

                
                gt_labels.append(gt_labels_j)
                gt_boxes.append(gt_boxes_j)
            
            gt_labels = torch.stack(gt_labels) # bs num_anchors
            gt_boxes = torch.stack(gt_boxes) # bs num_anchors 4

        # focal loss
        valid_mask = gt_labels >= 0

        # classifications = F.sigmoid(classifications)
        valid_classifications = classifications[valid_mask] # num_valid * num_classes
        p = torch.sigmoid(valid_classifications)
        gt_labels_target = F.one_hot(gt_labels[valid_mask].long(), num_classes=self.nc + 1)[
            :, :-1
        ].to(valid_classifications.dtype)  # no loss for the last (background) class
        # bce_loss = F.binary_cross_entropy(valid_classifications, gt_labels_target, reduction="none")
        bce_loss = F.binary_cross_entropy_with_logits(valid_classifications, gt_labels_target, reduction="none")
        # import ipdb;ipdb.set_trace()
        p_t = p * gt_labels_target + (1 - p) * (1 - gt_labels_target)
        focal_weight = (1 - p_t) ** gamma
        cls_loss = bce_loss * focal_weight
        alpha_weight = gt_labels_target * alpha + (1 - alpha) * (1 - gt_labels_target)
        cls_loss = cls_loss * alpha_weight
        cls_loss = cls_loss.sum()

        # regression loss
        pos_mask = (gt_labels >= 0) & (gt_labels != self.nc) # bs num_anchors
        num_pos_anchors = pos_mask.sum().item()
        # if cls_loss / num_pos_anchors < 0.2:
        #     import ipdb;ipdb.set_trace()
        #     print('found')
        # transform gt to delta
        gt_ctr_x = 0.5 * (gt_boxes[:, :, 0] + gt_boxes[:, :, 2])
        gt_ctr_y = 0.5 * (gt_boxes[:, :, 1] + gt_boxes[:, :, 3])
        gt_width = gt_boxes[:, :, 2] - gt_boxes[:, :, 0]
        gt_height = gt_boxes[:, :, 3] - gt_boxes[:, :, 1]

        wx, wy, ww, wh = [1.0, 1.0, 1.0, 1.0]
        dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
        dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
        dw = ww * torch.log(gt_width / anchor_widths)
        dh = wh * torch.log(gt_height / anchor_heights) # bs num_anchors
        # print('dx', dx)
        # print('dy', dy)
        # print('dw', dw)
        # print('dh', dh)

        deltas = torch.stack((dx, dy, dw, dh), dim=2) # bs num_anchors 4

        reg_loss = smooth_l1_loss(
            regressions[pos_mask],
            deltas[pos_mask],
            0.1,
            'sum'
        )
        final_reg_loss = reg_loss / max(num_pos_anchors, 1)
        return {
            'cls_loss': cls_loss / max(num_pos_anchors, 1),
            'reg_loss': reg_loss / max(num_pos_anchors, 1),
        }




# def attempt_load(weights, map_location=None, inplace=True):
#     # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
#     model = ResnetFPN(50, 80).cuda()
#     ckpt = torch.load(weights[0], map_location=map_location)  # load
#     res = model.load_state_dict(ckpt['model'], strict=False)
#     print(res)
#     model = model.float().eval()
#     return model



def attempt_load(weights, map_location=None, inplace=True):
    from models.experimental import Ensemble

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # FP32 model

    # # Compatibility updates
    # for m in model.modules():
    #     if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
    #         m.inplace = inplace  # pytorch 1.7.0 compatibility
    #     # elif type(m) is Conv:
    #     #     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='retinanet.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # print(model)
    sample = torch.rand(4, 3, 640, 640).to(device)
    cls_id = torch.randint(0, 80,(8,1))
    xy = torch.randint(0, 400, (8, 2))
    wh = torch.randint(30, 120, (8, 2))
    labels = []
    for i in range(4):
        label = torch.cat([torch.ones(8, 1) * i, cls_id, xy, wh], dim=1).to(device)
        labels.append(label)
    labels = torch.cat(labels, dim=0)
    model.train()
    # normalize
    mean = torch.Tensor([103.530, 116.280, 123.675]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([57.375, 57.120, 58.395]).view(1, 3, 1, 1).to(device)
    sample = (sample - mean) / std
    pred, cls_preds, loc_preds = model(sample)
    anchors = model.anchors
    print('cls_preds: ', cls_preds.shape)
    print('loc_preds: ', loc_preds.shape)
    print('anchors', anchors.shape)
    losses = model.loss(cls_preds, loc_preds, anchors, labels)
    print(losses)