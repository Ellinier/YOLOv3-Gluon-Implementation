import argparse

import mxnet as mx
from mxnet import gluon

import gluoncv.data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

from yolo import yolo_416_darknet53_voc
from transform import YOLODefaultTrainTransform, YOLODefaultValTransform
from yolo_loss import YOLOLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train yolo networks.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help='Base network name which serves as feature extraction base.')
    parser.add_argument('--data-shape', type=int, default=416,
                        help='Input data shape, use 416.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger'
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epoches at which learning rate decays. default is 160,180.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    # with autograd.train_mode():
    #     _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    train_batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLODefaultTrainTransform(width, height, net)),
        batch_size, True, batchify_fn=train_batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLODefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def train(net, train_data, val_data, eval_metric, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}
    )

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])  # [160.0, 180.0]

    # assign loss
    yolo_loss = YOLOLoss()
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    obj_metrics = mx.metric.Loss('ObjLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)



models = {
    'yolo_416_darknet53_voc': yolo_416_darknet53_voc
    }

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo', str(args.data_shape), args.network, args.dataset))  # yolo_416_darknet53_voc
    net = models[net_name]()
    # print(net)

    # initialize parameters
    for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, args)
