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
from resnet_fpn import ResnetFPN
from fcos_utils import FCOSLabelTarget, iou_loss
import math
from utils.loss import smooth_l1_loss
from utils.general import make_divisible, check_file, set_logging, xyxy2xywh
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

from adet.utils.comm import compute_locations
from fcos_outputs import FCOSOutputs

class RegressionModel(nn.Module):
    def __init__(self, num_features_in):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, 4, kernel_size=3, padding=1)

        

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

        return out

        # # out is B x C x W x H, with C = 4*num_anchors
        # out = out.permute(0, 2, 3, 1) # bs h w c
        # return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=80):
        super(ClassificationModel, self).__init__()

        self.nc = num_classes

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(num_features_in, self.nc, kernel_size=3, padding=1)

        self.centerness = nn.Conv2d(num_features_in, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        feat = self.act4(out)

        out = self.output(feat)

        centerness = self.centerness(feat) # B 1 H W

        return out, centerness

        # out = out.permute(0, 2, 3, 1) # B H W C
        # out = out.view(x.shape[0], -1, self.nc).contiguous()
        # centerness = self.centerness(feat) # B 1 H W
        # return out, centerness.view(x.shape[0], -1, 1)
        # return out, torch.ones(x.shape[0], out.shape[1], 1).to(out.device)

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale

class Model(nn.Module):
    onnx_dynamic = False  # ONNX export parameter
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
        # FCOS hyp
        self.object_sizes_of_interest = self.yaml.get('object_sizes_of_interest', [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, 9999999],
        ]) # interest size for each feat map
        self.center_sampling_radius = self.yaml.get('center_sampling_radius', -1)

        super(Model, self).__init__()
        self.backbone = ResnetFPN(depth, fpn_fs, pretrained=pretrained)
        self.classificationModel = ClassificationModel(fpn_fs, num_classes=self.nc)
        self.regressionModel = RegressionModel(fpn_fs)
        
        # init stride
        stride = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.register_buffer('stride', stride.clone())
        self.nl = self.stride.shape[0] # number of detection layers
        assert self.nl == len(self.object_sizes_of_interest), f'nl: {self.nl} != len(object_sizes_of_interest): {len(self.object_sizes_of_interest)}'

        # construct scales
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.nl)])

        # init (x,y) grid 
        self.grid = [torch.zeros(1)] * self.nl  # init grid

        # init fcos target
        # self.fcos_target = FCOSLabelTarget(self.stride, self.object_sizes_of_interest, self.nc, self.center_sampling_radius)
        self.fcos_outputs = FCOSOutputs()

        for modules in [self.classificationModel, self.regressionModel]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        prior = 0.01
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, -1, 2)).float()

    @property
    def device(self):
        return self.stride.device

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    # def forward(self, imgs, **kwargs):
    #     # shape
    #     _, _, H, W = imgs.shape
    #     features = self.backbone(imgs)
    #     loc_preds = []
    #     cls_preds = []
    #     centerness = []
    #     inference_out = []
    #     # import ipdb;ipdb.set_trace()
    #     for i, feat in enumerate(features):
    #         ny, nx = feat.shape[2:4]
    #         loc_pred = torch.exp(self.scales[i](self.regressionModel(feat)))
    #         logits, cent = self.classificationModel(feat)
    #         # import ipdb;ipdb.set_trace()
    #         loc_preds.append(loc_pred) # bs nx*ny 4
    #         cls_preds.append(logits) # bs nx*ny nc
    #         centerness.append(cent) # bs nx*ny 1
    #         # update grid
    #         # import ipdb;ipdb.set_trace()
    #         with torch.no_grad():
    #             if self.grid[i].shape[2:4] != feat.shape[2:4] or self.onnx_dynamic:
    #                 self.grid[i] = self._make_grid(nx, ny).to(feat.device) # (1, -1, 2)
    #                 self.grid[i] = (self.grid[i] + 0.5) * self.stride[i]
    #         if not self.training:
    #             # inference cls
    #             cls_score = logits.sigmoid()
    #             center_score = cent.sigmoid()
    #             cls_score *= cent
    #             # inference reg
    #             xy1s = self.grid[i].to(loc_pred.device) - loc_pred[...,:2]
    #             xy2s = self.grid[i].to(loc_pred.device) + loc_pred[..., 2:]
    #             pred = torch.cat([xy1s, xy2s, cls_score], -1) # bs nx*ny 4+nc
    #             pred[..., :4] = xyxy2xywh(pred[..., :4])
    #             inference_out.append(pred)
    #             inference_out = torch.cat(inference_out, dim=1)
    #     return inference_out, cls_preds, loc_preds, centerness
    
    def forward(self, imgs, **kwargs):
        # shape
        _, _, H, W = imgs.shape
        features = self.backbone(imgs)
        loc_preds = []
        cls_preds = []
        centerness = []
        inference_out = []
        locations = self.compute_locations(features) # list of shape (np, 2)
        # import ipdb;ipdb.set_trace()
        for i, feat in enumerate(features):
            loc_pred = torch.exp(self.scales[i](self.regressionModel(feat)))
            logits, cent = self.classificationModel(feat)
            # import ipdb;ipdb.set_trace()
            loc_preds.append(loc_pred) # bs 4 nx ny
            cls_preds.append(logits) # bs nc nx ny
            centerness.append(cent) # bs 1 nx ny
        
        if self.training:
            gt_instances = kwargs['targets']
            results, losses = self.fcos_outputs.losses(
                cls_preds, loc_preds, centerness,
                locations, gt_instances
            )
            return losses
        else:
            for level, (loc_pred, cls_pred, cent) in enumerate(zip(loc_preds, cls_preds, centerness)):
                loc_pred = loc_pred.permute(0, 2, 3, 1).view(loc_pred.shape[0], -1, 4)
                cls_pred = cls_pred.permute(0, 2, 3, 1).view(cls_pred.shape[0], -1, self.nc)
                cent = cent.permute(0, 2, 3, 1).view(cent.shape[0], -1, 1)
                # import ipdb;ipdb.set_trace()
                xy1 = locations[level][None] - loc_pred[:,:,:2]
                xy2 = locations[level][None] + loc_pred[:,:,2:]
                cls_score = cls_pred.sigmoid() * cent.sigmoid()
                # import ipdb;ipdb.set_trace()
                pred = torch.cat([xy1, xy2, cls_score], dim=2)
                pred[..., :4] = xyxy2xywh(pred[...,:4])
                inference_out.append(pred)
            inference_out = torch.cat(inference_out, dim=1)
            return inference_out

    
    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.stride[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


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
    parser.add_argument('--cfg', type=str, default='fcos.yaml', help='model.yaml')
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

    # pred, cls_preds, loc_preds, centerness = model(sample)
    # grid = model.grid
    # print('cls_preds: ', cls_preds[0].shape)
    # print('loc_preds: ', loc_preds[0].shape)
    # print('centerness', centerness[0].shape)
    # print('grid', grid[0].shape)
    # losses = model.loss(cls_preds, loc_preds, centerness, labels)
    # print(losses)
    loss = model(sample, targets=labels)
    print(loss)