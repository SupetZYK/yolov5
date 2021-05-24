import torch
import math

class Anchors:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.strides = [2 ** (i+3) for i in range(len(self.anchor_areas))] # stride of each fm
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()
        self.offset = 0.0
    
    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.
        Returns:
            anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            # for sr in self.scale_ratios:
            for ar in self.aspect_ratios:  # h/w = ar
                w = math.sqrt(s/ar)
                h = ar * w
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)
    
    def _get_anchor_boxes(self, input_size, device='cuda', anchor_format='xywh'):
        '''Compute anchor boxes for each feature map.
        Args:
            input_size: (tensor) model input size of (w,h).
        Returns:
            boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell. In format [ctr_x, ctr_y, w, h]s
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(torch.tensor(input_size).float() / torch.tensor(stride)).ceil() for stride in self.strides] #p3 --> p7 feature map sizes
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            y, x = torch.meshgrid(torch.arange(fm_h, device=device),torch.arange(fm_w, device=device))
            xy = torch.stack([x,y], dim=2) + self.offset  # [fm_h fm_w, 2]
            xy = (xy*self.strides[i]).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].to(device).view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(1,-1,4))
        res = torch.cat(boxes, 1)
        if anchor_format == 'xywh':
            return res
        elif anchor_format == 'xyxy':
            return self.xywh2xyxy(res)
        else:
            raise ValueError()
    
    def predict_transform(self, loc_preds, anchor_boxes, clip=True, bbx_reg_weight=[1, 1, 1, 1]):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [bs, #anchors, 4].
          input_size: (int/tuple) model input size of (w,h).
          anchor_boxes: format xywh
        Returns:
          boxes: (tensor) decode box locations, sized [bs,#obj,4].
        '''
        # if anchor_boxes is None:
        #     input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
        #                 else torch.Tensor(input_size)
        #     anchor_boxes = self._get_anchor_boxes(input_size).to(loc_preds.device)
        # import ipdb;ipdb.set_trace()
        bbx_reg_weight = torch.Tensor(bbx_reg_weight).view(1,1, -1).to(loc_preds.device)
        loc_preds = loc_preds / bbx_reg_weight
        loc_xy = loc_preds[...,:2]
        loc_wh = loc_preds[...,2:]
        xy = loc_xy * anchor_boxes[...,2:] + anchor_boxes[...,:2]
        wh = loc_wh.exp() * anchor_boxes[...,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], -1)  # [#anchors,4]
        # if clip:
        #     boxes[:,0::2] = torch.clamp(boxes[:,0::2], min=0, max=input_size[0])
        #     boxes[:,1::2] = torch.clamp(boxes[:,1::2], min=0, max=input_size[1])
        return boxes
    
    @classmethod
    def xywh2xyxy(cls, anchors):
        ''' anchor format transform
        Args:
            anchors: shape [#anchors, 4]
        '''
        xy = anchors[..., :2]
        wh = anchors[..., 2:]
        return torch.cat([xy - wh/2, xy + wh/2], -1)
    
    @classmethod
    def xyxy2xywh(cls, anchors):
        ''' format transform
        Args:
            anchors: shape [#anchors, 4]
        '''
        wh = anchors[...,2:] - anchors[..., :2]
        xy = anchors[..., :2] + wh / 2
        return torch.cat([xy, wh], -1)


def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        # ids = (ovr<=threshold).nonzero().squeeze()
        ids = torch.where(ovr<=threshold)[0]
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
