"""Yolov3 network for object detection."""
from __future__ import division


from mxnet import nd
from mxnet import autograd
from mxnet import symbol
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock


from Darknet53 import darknet53_416
from Darknet53 import Convolutional

from gluoncv.data import VOCDetection

# def Upsample(data, stride):
# 	return nd.Upsample(data, scale=stride, sample_type='nearest')


def upsample(x, stride=2):
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


class FeatureExpander(HybridBlock):
    """YOLO Feature Expander.

    Parameters
    ----------
    channels: int
        Number of channels for the first convolutional layer on feature expander block.

    """
    def __init__(self, channels):
        super(FeatureExpander, self).__init__()
        self.channels = channels

        with self.name_scope():
            self.conv_stack = nn.HybridSequential()
            for i in range(5):
                if i % 2 == 0:
                    self.conv_stack.add(Convolutional(self.channels, 1, 1, 0))
                else:
                    self.conv_stack.add(Convolutional(self.channels * 2, 3, 1, 1))
            self.single_layer = Convolutional(self.channels*2, 3, 1, 1)
            self.transition = Convolutional(int(self.channels/2), 1, 1, 0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv_stack(x)
        out = self.single_layer(out)
        out = self.transition(out)
        sub_x = upsample(out)
        return out, sub_x


class YOLOLayer(HybridBlock):
    """YOLO Detection network.

    Parameters
    ----------
    anchors: iterable of list
        Scales of anchors in each output feature map in form of [width, height].
    stride: int
        Stride of grid in each output feature map.
    num_classes: int
        Number of categories.

    """
    def __init__(self, anchors, stride, num_classes):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors                         # [[10,13], [16,30], [33,23]]
        self.stride = stride                           # 8
        self.num_anchors = len(anchors)                # 3
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes              # 25

        self.prediction = Convolutional(self.bbox_attrs*self.num_anchors, 1, 1, 0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.prediction(x)
        out = out.transpose((0, 2, 3, 1))                                # (B, H, W, (4+1+n_Classes)*n_anchor)
        out = out.reshape((0, 0, 0, self.num_anchors, self.bbox_attrs))  # (B, H, W, n_anchor, (4+1+n_Classes))

        if autograd.is_training():
            return out.reshape((0, -1, self.bbox_attrs))

        # Get prediction
        tx = F.sigmoid(out.slice_axis(begin=0, end=1, axis=-1))            # Center x
        ty = F.sigmoid(out.slice_axis(begin=1, end=2, axis=-1))            # Center y
        tw = out.slice_axis(begin=2, end=3, axis=-1)                       # Width
        th = out.slice_axis(begin=3, end=4, axis=-1)                       # Height
        objness = F.sigmoid(out.slice_axis(begin=4, end=5, axis=-1))       # Conf
        cls_pred = F.sigmoid(out.slice_axis(begin=5, end=None, axis=-1))   # Cls pred

        b, h, w, n, s = tx.shape

        # Calculate offsets for each grid
        grid_x = nd.tile(nd.arange(0, w, repeat=(n * 1), ctx=tx.context).reshape((1, 1, w, n, 1)), (b, h, 1, 1, 1))
        grid_y = nd.tile(nd.arange(0, h, repeat=(w * n * 1), ctx=ty.context).reshape((1, h, w, n, 1)), (b, 1, 1, 1, 1))
        scaled_anchors_w = [a_w/self.stride for a_w, _ in self.anchors]
        scaled_anchors_h = [a_h/self.stride for _, a_h in self.anchors]
        anchor_w = nd.tile(nd.array(scaled_anchors_w).reshape((1, 1, 1, -1, 1)), (b, h, w, 1, 1))
        anchor_h = nd.tile(nd.array(scaled_anchors_h).reshape((1, 1, 1, -1, 1)), (b, h, w, 1, 1))

        # anchor = F.broadcast_mul(F.concat(grid_x, grid_y, anchor_w, anchor_h, dim=-1), self.stride)  # anchor relative to img

        # if autograd.is_training():
        #     return out.reshape((0, -1, self.bbox_attrs)), anchor.reshape((0, -1, 4))

        # Add offset and scale with anchors
        bx = tx + grid_x
        by = ty + grid_y
        bw = F.broadcast_mul(F.exp(tw), anchor_w)
        bh = F.broadcast_mul(F.exp(th), anchor_h)
        # boxes_pred = F.concat(bx, by, bw, bh, dim=-1)

        # convert to corner format boxes
        half_w = bw / 2
        half_h = bh / 2
        left = F.clip(bx - half_w, 0, 1)
        top = F.clip(by - half_h, 0, 1)
        right = F.clip(bx + half_w, 0, 1)
        bottom = F.clip(by + half_h, 0, 1)
        boxes_pred = F.concat(left, top, right, bottom, dim=-1)

        # cls_score = F.broadcast_mul(objness, cls_pred)
        # cid = nd.argmax(cls_score, axis=-1, keepdims=True)

        # output = F.concat(*[cid, cls_score, left, top, right, bottom], dim=4)

        # output = F.concat(boxes_pred.reshape(b, -1, 4)*self.stride,
        #                   objness.reshape(b, -1, 1),
        #                   cls_pred.reshape(b, -1, self.num_classes),
        #                   dim=-1)
        #
        # return output

        boxes_pred = boxes_pred.reshape(b, -1, 4)*self.stride
        objness = objness.reshape(b, -1, 1)
        cls_pred = cls_pred.reshape(b, -1, self.num_classes)

        return boxes_pred, objness, cls_pred


class YOLO(HybridBlock):
    """YOLO network for yolov3.

    Parameters
    ----------
    features: HybridBlock
        HybridBlock which generate multiple outputs for prediction.
    feat_expand: bool
        Expand feature maps of not.
    classes: iterable of str
        Names of categories.
    anchors: iterable of list
        Scales of anchors for each output feature map.
    strides: list of int
        Strides of grid for each output feature map.
    nms_thresh: float, default is o.45
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk: int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every detection
        result is used in NMS.
    post_nms: int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, features, feat_expand, classes, anchors, mask, strides,
                 nms_thresh=0.45, nms_topk=400, post_nms=100):
        super(YOLO, self).__init__()
        self.features = features()
        self.feat_expand = feat_expand
        self.classes = classes
        self.num_classes = len(self.classes)
        self.anchors = anchors
        self.mask = mask
        self.masked_anchors = [[self.anchors[i] for i in n] for n in self.mask]
        # masked_anchors=[[[116, 90], [156, 198], [373, 326]],
        #                 [[30, 61], [62, 45], [59, 119]],
        #                 [[10, 13], [16, 30], [33, 23]]]
        self.strides = strides  # [8, 16, 32]
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            if self.feat_expand:
                self.channels = [128, 256, 512]
                self.feature_expander = nn.HybridSequential()
                for channels in self.channels:
                    self.feature_expander.add(FeatureExpander(channels))
            self.detection = nn.HybridSequential()
            for anchor, stride in zip(self.masked_anchors, self.strides):  # self.anchor: [[10,13], [16,30], [33,23]]
                self.detection.add(YOLOLayer(anchor, stride, self.num_classes))

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
            result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None
        """
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def hybrid_forward(self, F, x, *args, **kwargs):
        # featmap_1, featmap_2, featmap_3 = self.features(x)
        # featmap = [featmap_1, featmap_2, featmap_3]
        featmap = self.features(x)
        if self.feat_expand:
            for i, expander in zip(range(len(featmap))[::-1], self.feature_expander[::-1]):
                featmap[i], sub_feat = expander(featmap[i])
                if i != 0:
                    featmap[i-1] = F.concat(featmap[i-1], sub_feat)

        if autograd.is_training():
            output = []
            # anchors = []
            for feat, detection in zip(featmap, self.detection):
                out = detection(feat)
                output.append(out)
                # anchors.append(anchor)

            predictions = F.concat(*output, dim=1)
            # default_anchors = F.concat(*anchors, dim=1)
            if autograd.is_recording():
                return predictions
            return (predictions, self.anchors, self.mask, self.strides)

        boxes_preds = []
        objness_scores = []
        cls_preds = []
        for feat, detection in zip(featmap, self.detection):
            boxes_pred, objness, cls_pred = detection(feat)
            boxes_preds.append(boxes_pred)
            objness_scores.append(objness)
            cls_preds.append(cls_pred)

        tboxes_preds, tcls_scores, tcls_ids = yolo2target(boxes_preds, objness_scores, cls_preds)
        # cls_scores = F.broadcast_mul(objness_scores, cls_preds)
        # cls_id = F.argmax(cls_scores, axis=-1, keepdims=True)
        # scores = F.pick(cls_scores, cls_id, axis=-1)

        return tboxes_preds, tcls_scores, tcls_ids


def get_yolo(features, feature_expand, classes, anchors, mask, strides, pretrained):
    """Get YOLO models.

    Parameters
    ----------
    features: HybridBlock
        HybridBlock which generate multiple outputs for prediction.
    feature_expand: bool
        Expand feature maps of not.
    classes: iterable of str
        Names of categories.
    anchors: iterable of list
        Scales of anchors for each output feature map.
    strides: list of int
        Strides of grid for each output feature map.
    pretrained: bool
        load pretrained parameters or not.

    Returns
    -------
    HybridBlock
        A YOLO detection network.
    """
    net = YOLO(features, feature_expand, classes, anchors, mask, strides)
    if pretrained:
        pass
    return net


def yolo_416_darknet53_voc(pretrained=False):
    """YOLO architecture with darknet53 for Pascal VOC.

    Parameters
    ----------

    Returns
    -------
    HybridBlock
        A YOLO detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_yolo(features=darknet53_416, feature_expand=True, classes=classes,
                   # anchors=[[[10, 13], [16, 30], [33, 23]],
                   #          [[30, 61], [62, 45], [59, 119]],
                   #          [[116, 90], [156, 198], [373, 326]]],
                   # anchors=[[[116, 90], [156, 198], [373, 326]],
                   #          [[30, 61], [62, 45], [59, 119]],
                   #          [[10, 13], [16, 30], [33, 23]]],
                   anchors=[[10, 13], [16, 30], [33, 23],
                            [30, 61], [62, 45], [59, 119],
                            [116, 90], [156, 198],[373, 326]],
                   mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                   strides=[32, 16, 8], pretrained=pretrained)
    return net
