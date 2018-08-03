from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import loss


def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr


class YOLOLoss(Block):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.sigmoid_bce_loss = loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.l2_loss = loss.L2Loss()

    def forward(self, center_preds, scale_preds, obj_preds, cls_preds,
                center_targets, scale_targets, obj_targets, cls_targets):
        center_preds, scale_preds, obj_preds, cls_preds, center_targets, scale_targets, obj_targets, cls_targets = [_as_list(x) \
                for x in (center_preds, scale_preds, obj_preds, cls_preds, center_targets, scale_targets, obj_targets, cls_targets)]

        # compute element-wise cross entropy loss and sort
        center_losses = []
        scale_losses = []
        objness_losses = []
        cls_losses = []
        sum_losses = []

        for cp, sp, op, cp, ct, st, ot, ct in zip(*[center_preds, scale_preds, obj_preds, cls_preds, center_targets, scale_targets, obj_targets, cls_targets]):
            objness_loss = self.sigmoid_bce_loss(op, ot, ot >= 0)
            objness_losses.append(objness_loss)
            # TODO: add weight to center_loss
            center_loss = self.sigmoid_bce_loss(cp, ct)
            center_losses.append(center_loss)
            # TODO: add weight to scale_loss
            scale_loss = self.l2_loss(sp, st)
            scale_losses.append(scale_loss)
            # TODO: add weight to cls_loss
            cls_loss = self.sigmoid_bce_loss(cp, ct)
            cls_losses.append(cls_loss)
            sum_loss = objness_loss + center_loss + scale_loss + cls_loss
            sum_losses.append(sum_loss)

        return center_losses, scale_losses, objness_losses, cls_losses, sum_losses
