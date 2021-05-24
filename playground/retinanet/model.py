# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from torch import nn
import torch.nn.functional as F
import math
from models.common import autoShape
from resnet_fpn import ResnetFPN
from utils.loss import smooth_l1_loss
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging, pairwise_iou
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.all_anchors = [torch.zeros(1)] * self.nl # init anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    
    @property
    def device(self):
        return self.anchor_grid.device

    def forward(self, x):
        # x: [features]
        # x = x.copy()  # for profiling
        z = []  # inference output
        xs = [] # raw output
        # import ipdb;ipdb.set_trace()
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,84)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #(bs,3,20,20,84)
            xs.append(x[i].view(x[i].shape[0], -1, self.no))
            if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                self.grid[i] = self._make_grid(nx, ny).to(self.anchor_grid.device) # (1,1,20,20,2)
                # refresh anchors
                self.all_anchors[i] = torch.cat([(self.grid[i].repeat(1,self.na,1,1,1) + 0.5) * self.stride[i], self.anchor_grid[i].repeat(1,1,ny,nx,1)], dim=4) # (1,na,ny,nx,4)
                self.all_anchors[i] = self.all_anchors[i].view(1, -1, 4)
            y = x[i].clone().view(bs, -1, self.no)
            y[...,4:] = y[...,4:].sigmoid()
            # y = y.sigmoid()
            if self.inplace:
                y[..., 0:2] = y[..., 0:2] * self.all_anchors[i][..., 2:] + self.all_anchors[i][..., :2]
                y[..., 2:4] = torch.exp(y[..., 2:4]) * self.all_anchors[i][..., 2:]
            else:
                xy = y[..., 0:2] * self.all_anchors[i][..., 2:] + self.all_anchors[i][..., :2]
                wh = y[..., 2:4].exp() * self.all_anchors[i][..., 2:]
                y = torch.cat([xy, wh, y[..., 4:]], -1)
            z.append(y)
        return (torch.cat(z, 1), torch.cat(xs, 1)) if self.training else (torch.cat(z, 1), torch.cat(xs, 1))
    
    # def get_anchors(self, img_h, img_w, ):
    #     all_anchors = []
    #     for i in range(self.nl):
    #         grid = self._make_grid(nx, ny)
    #         tmp_anchors = torch.cat([(grid.repeat(1,na,1,1,1) + 0.5) * self.stride[i], self.anchor_grid.repeat(1,1,ny,nx,1)], dim=4) # (1,na,nx,ny,4)
    #         all_anchors.append(tmp_anchors.view(-1, 4))
    #     return all_anchors

    def loss(self, all_anchors, nn_raw_out, targets, **kwargs):
        """ losses for retinanet

        Args:
            all_anchors (list[tensor]): List[layer_anchors],layer_anchors shape (-1,4), xywh
            nn_raw_out (tensor): nn output, (bs, N_a, 84), no sigmoid applied
            targets (tensor): list[label] label shape (-1, 6), [img_id,cls_id,x,y,w,h]
        """
        alpha = 0.25
        gamma = 2.0
        num_images = nn_raw_out.shape[0]
        anchors = torch.cat(all_anchors, dim=1).view(-1, 4)

        regressions = nn_raw_out[..., 0:4]
        classifications = nn_raw_out[...,4:]

        # match anno to anchors first
        with torch.no_grad():
            gt_labels = []
            gt_boxes = []
            for j in range(num_images):
                target = targets[targets[:,0]==j]
                bbox_annotation = target[:,2:]
                anno_label = target[:,1]
                # import ipdb;ipdb.set_trace()
                if bbox_annotation.shape[0] == 0:
                    gt_labels_j = torch.zeros(classifications[j].shape[0]).to(self.device) + self.nc # num_anchors
                    gt_boxes_j = torch.zeros_like(anchors) # num_anchors, 4
                else:
                    IoU = pairwise_iou(anchors, bbox_annotation, x1y1x2y2=False) # num_anchors x num_annotations
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
            #--- end of match
        # focal loss
        valid_mask = gt_labels >= 0
        valid_classifications = classifications[valid_mask] # num_valid * num_classes
        p = torch.sigmoid(valid_classifications)
        gt_labels_target = F.one_hot(gt_labels[valid_mask].long(), num_classes=self.nc + 1)[
            :, :-1
        ].to(valid_classifications.dtype)  # no loss for the last (background) class
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
        # get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        # if cls_loss / num_pos_anchors < 0.2:
        #     import ipdb;ipdb.set_trace()
        #     print('found')

        gt_ctr_x = gt_boxes[:, :, 0]
        gt_ctr_y = gt_boxes[:, :, 1]
        gt_widths = gt_boxes[:, :, 2]
        gt_heights = gt_boxes[:, :, 3]

        anchor_ctr_x = anchors[:, 0].view(1, -1)
        anchor_ctr_y = anchors[:, 1].view(1, -1)
        anchor_widths = anchors[:, 2].view(1, -1)
        anchor_heights = anchors[:, 3].view(1, -1)

        wx, wy, ww, wh = [1.0, 1.0, 1.0, 1.0]
        dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
        dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
        dw = ww * torch.log(gt_widths / anchor_widths)
        dh = wh * torch.log(gt_heights / anchor_heights) # bs num_anchors
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

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        fpn_fs = self.yaml.get('fpn_feat_size', 256)
        self.model = ResnetFPN(self.yaml.get('depth', 50), fpn_feat_size=fpn_fs, pretrained=True)
        self.detect = Detect(self.yaml['nc'], self.yaml['anchors'], ch=(fpn_fs, fpn_fs, fpn_fs))

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.detect
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.model.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # print('no init bias')
            # self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        for layer in self.detect.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        
        self._initialize_biases()
        initialize_weights(self)
        self.info()
        logger.info('')


    def forward(self, x, augment=False, profile=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False):
        return self.detect(self.model(x))
        

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            # b.data[:, 4:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b.data[:, 4:] = - math.log((1-0.01) / (0.01))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.detect
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    # def nms(self, mode=True):  # add or remove NMS module
    #     present = type(self.model[-1]) is NMS  # last layer is NMS
    #     if mode and not present:
    #         logger.info('Adding NMS... ')
    #         m = NMS()  # module
    #         m.f = -1  # from
    #         m.i = self.model[-1].i + 1  # index
    #         self.model.add_module(name='%s' % m.i, module=m)  # add
    #         self.eval()
    #     elif not mode and present:
    #         logger.info('Removing NMS... ')
    #         self.model = self.model[:-1]  # remove
    #     return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def attempt_load(weights, map_location=None, inplace=True):
    from models.experimental import Ensemble

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        # elif type(m) is Conv:
        #     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

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
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)

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
    infer_out, nn_out = model(sample)
    print('infer out: ', infer_out.shape)
    print('nn_out: ', nn_out.shape)
    print('anchors', torch.cat(model.detect.all_anchors, dim=1).shape)
    losses = model.detect.loss(model.detect.all_anchors, nn_out, labels)
    print(losses)
    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
