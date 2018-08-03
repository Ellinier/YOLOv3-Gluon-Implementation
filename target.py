from mxnet import nd
from mxnet.gluon import Block

from gluoncv.nn.bbox import BBoxCornerToCenter

class YOLOTargetGenerator(Block):
    """Training targets generator for YOLO Object Detection.

        Parameters
        ----------
        iou_thresh : float
            IOU overlap threshold for maximum matching, default is 0.5.
        neg_thresh : float
            IOU overlap threshold for negative mining, default is 0.5.
        negative_mining_ratio : float
            Ratio of hard vs positive for negative mining.
        stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
            Std value to be divided from encoded values.
    """
    def __init__(self, num_classes, **kwargs):
        super(YOLOTargetGenerator, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)

    def forward(self, img_size, gt_boxes, gt_ids):

        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = len(anchor_mask)
        grid_shapes = [13, 26, 52]
        center_targets = [nd.zeros((grid_shapes[i], grid_shapes[i], len(anchor_mask[i]), 2),
                           dtype='float32') for i in range(num_layers)]
        scale_targets = [nd.zeros((grid_shapes[i], grid_shapes[i], len(anchor_mask[i]), 2),
                                   dtype='float32') for i in range(num_layers)]
        obj_targets = [nd.zeros((grid_shapes[i], grid_shapes[i], len(anchor_mask[i]), 1),
                                  dtype='float32') for i in range(num_layers)]
        cls_targets = [nd.ones((grid_shapes[i], grid_shapes[i], len(anchor_mask[i]), self.num_classes),
                                  dtype='float32')*-1 for i in range(num_layers)]

        # print gt_boxes.shape  # (1L, n_label, 4L)
        gtx, gty, gtw, gth = self.bbox2center(gt_boxes)  # gtx: (1L, n_label, 1L)
        shift_gt_boxes = nd.transpose(nd.concat(gtx * 0, gty * 0, gtw, gth, dim=-1), (1, 0, 2))  # (n_label, 1, 4)
        anchors = nd.array(anchors).reshape((0, -1, 2))
        shift_anchor_boxes = nd.concat(0 * anchors, anchors, dim=-1).reshape((1, -1, 4))  # zero center anchor boxes (1L, n_anchor, 4L)

        # compute iou
        ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes, format='center')  # (1, n_anchor, n_label, 1)
        matches = ious.argmax(axis=1).asnumpy()  # (1, n_label, 1)
        for i, m in enumerate(matches.flatten()):
            for x in anchor_mask:
                if m in x:
                    layer_index = anchor_mask.index(x)
                    inlayer_index = x.index(int(m))
                    mask_wh = anchors[int(m)]

            grid_size = grid_shapes[layer_index]
            stride = img_size/grid_size
            cx = nd.floor(gtx/stride)
            cy = nd.floor(gty/stride)
            tx = gtx/stride-cx
            ty = gty/stride-cy
            tw = nd.log(gtw/mask_wh[:, 0])
            th = nd.log(gth/mask_wh[:, 1])
            center_targets[layer_index][cx, cy, inlayer_index, 0] = tx
            center_targets[layer_index][cx, cy, inlayer_index, 1] = ty
            scale_targets[layer_index][cx, cy, inlayer_index, 0] = tw
            scale_targets[layer_index][cx, cy, inlayer_index, 1] = th
            obj_targets[layer_index][cx, cy, inlayer_index, 0] = 1
            cls_targets[layer_index][cx, cy, inlayer_index, gt_ids[:, i]] = 1

        center_targets = [center_target.reshape((-1, 2)) for center_target in center_targets]
        center_targets = nd.concat(*center_targets, dim=0)
        scale_targets = [scale_target.reshape((-1, 2)) for scale_target in scale_targets]
        scale_targets = nd.concat(*scale_targets, dim=0)
        obj_targets = [obj_target.reshape((-1, 1)) for obj_target in obj_targets]
        obj_targets = nd.concat(*obj_targets, dim=0)
        cls_targets = [cls_target.reshape((-1, self.num_classes)) for cls_target in cls_targets]
        cls_targets = nd.concat(*cls_targets, dim=0)

        return center_targets, scale_targets, obj_targets, cls_targets
