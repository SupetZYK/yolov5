import torch
from utils.general import xywh2xyxy, xyxy2xywh

INF = 999999

class FCOSLabelTarget(object):
    def __init__(self, strides, object_sizes_of_interest, nc, center_sampling_radius=-1, norm_reg_targets=False, **kwargs):
        self.strides = strides
        self.object_sizes_of_interest = object_sizes_of_interest
        self.center_sampling_radius = center_sampling_radius
        self.norm_reg_targets = norm_reg_targets
        self.nc = nc
        self.kwargs = kwargs

    def prepare_targets(self, points, targets):
        """
        Args:
            points: List(tensor), each tensor for grid point of a feature map, shape (1, n, 2)
            targets: List(tensor)
        """
        # cat all grid points and match with targets
        # first, we should attach size limit to each grid point
        device = targets[0].device
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = torch.Tensor(self.object_sizes_of_interest[l]).to(device)
            # import ipdb;ipdb.set_trace()
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level.view(1, 2).expand(points_per_level.shape[1], 2)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [points_per_level.shape[1] for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points = [p.to(device) for p in points]
        points_all_level = torch.cat(points, dim=1).view(-1, 2)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)): # iter batch dim, i'th image
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = [] # each shape (bs, npoints)
        reg_targets_level_first = [] # each shape (bs, npoints, 4)
        for level in range(len(points)):
            labels_level_first.append(
                torch.stack([labels_per_im[level] for labels_per_im in labels])
            )
            # import ipdb;ipdb.set_trace()
            reg_targets_per_level = torch.stack([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ])

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, points_all, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = points_all[:, 0], points_all[:, 1]

        for j in range(len(targets)):
            target = targets[j]
            if target.shape[0] == 0:
                labels.append(torch.zeros(xs.shape[0]).to(target.device) + self.nc)
                reg_targets.append(torch.zeros(xs.shape[0], 4).to(target.device))
                continue
            # assert targets_per_im.mode == "xyxy"
            bboxes = target[:,1:] # xywh
            area = bboxes[:,2] * bboxes[:,3]
            bboxes = xywh2xyxy(bboxes)
            labels_per_im = target[:,0]

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the points within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(points_all), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # debug, draw points
            if 'debug_images' in self.kwargs:
                imgs = self.kwargs['debug_images']
                # get positive mask
                pos_mask = is_in_boxes & is_cared_in_the_level # npoints nboxes
                # pos_mask = is_in_boxes
                pos_mask = pos_mask.any(dim=1) # npoints
                # split into levels
                pos_masks = torch.split(pos_mask, self.num_points_per_level, dim=0)
                point_all_levels = torch.split(points_all, self.num_points_per_level, dim=0)
                for level, (p_mask, p) in enumerate(zip(pos_masks, point_all_levels)):
                    self._draw(imgs[j], self.strides[level], bboxes, p[p_mask], [], prefix=f'img_{j}')

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(points_all)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.nc
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets
    
    def _draw(self, img, stride, gtboxes, pos_points, ignore_points, prefix):
        img_show = img.copy()
        for i in range(len(gtboxes)):
            x1, y1, x2, y2 = list(map(int, gtboxes[i][:4]))
            color = (0, 255, 0)
            cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 2)

        for x, y in pos_points:
            cv2.circle(img_show, (int(x), int(y)), 2, (0, 0, 255), 2)
        for x, y in ignore_points:
            cv2.circle(img_show, (int(x), int(y)), 2, (255, 255, 255), 2)

        cv2.imwrite('{}_stride{}_info.jpg'.format(prefix, stride), img_show)

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

# def iou_loss(preds,targets):
#     '''
#     Args:
#     preds: [n,4] ltrb
#     targets: [n,4]
#     '''
#     lt=torch.min(preds[:,:2],targets[:,:2])
#     rb=torch.min(preds[:,2:],targets[:,2:])
#     wh=(rb+lt).clamp(min=0)
#     overlap=wh[:,0]*wh[:,1]#[n]
#     area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
#     area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
#     iou=overlap/(area1+area2-overlap + 1e-6)
#     loss=-iou.clamp(min=1e-6).log()
#     return loss.sum()
    

def iou_loss(pred, target, weight=None, loss_type='iou'):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                    (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion
    if loss_type == 'iou':
        losses = -torch.log(ious.clamp(min=1e-6))
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    elif loss_type == 'giou':
        losses = 1 - gious
    else:
        raise NotImplementedError

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        # assert losses.numel() != 0
        return losses.sum()

if __name__ == "__main__":
    import cv2
    from utils.general import scale_coords
    from utils.plots import plot_one_box
    from utils.datasets import letterbox
    from model import Model

    # load test image and target
    img0 = cv2.imread('/data/dataset/coco/val2017/000000000139.jpg')
    H, W = img0.shape[:2]
    targets = torch.Tensor([
        [62, 0.122913, 0.508222, 0.245826, 0.237745],
        [56, 0.629217, 0.630818, 0.0796916, 0.222307],
        [56, 0.49951, 0.636243, 0.0893758, 0.222362],
    ])
    targets[:,[1,3]] *= W
    targets[:,[2,4]] *= H
    targets[:,1:] = xywh2xyxy(targets[:,1:])

    # scale img and target
    img, ratio, pad = letterbox(img0)
    targets[:,1:] *= ratio[0]
    targets[:,[1,3]] += pad[0]
    targets[:,[2,4]] += pad[1]

    img_draw = img.copy()
    for target in targets:
        plot_one_box(target[1:], img_draw)
    cv2.imwrite('test.jpg', img_draw)

    targets[:,1:] = xyxy2xywh(targets[:,1:])

    # make grid for img
    strides = [8, 16, 32, 64, 128]
    points_all = []
    for stride in strides:
        ny = img.shape[0] // stride
        nx = img.shape[1] // stride
        points_all.append((Model._make_grid(nx,ny) + 0.5) * stride)
    
    # init fcos_target
    fcos_target = FCOSLabelTarget(
        strides=[8, 16, 32, 64, 128], 
        object_sizes_of_interest=[[-1, 64],
                                    [64, 128],
                                    [128, 256],
                                    [256, 512],
                                    [512, 9999999]],
        nc=80,
        debug_images=[img],
        center_sampling_radius=2,
    )

    # forward
    labels_level, regs_level = fcos_target.prepare_targets(points_all, [targets])
    # import ipdb;ipdb.set_trace()
    print('done')
