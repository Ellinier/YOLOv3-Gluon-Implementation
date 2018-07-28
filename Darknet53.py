"""Darknet53 network for object detection."""

from mxnet.gluon import nn
from mxnet.gluon import HybridBlock


def Convolutional(channels, kernel_size, strides, padding, batch_norm=True):
    """Convolutional layer in yolov3
    Conv2D+BatchNorm+LeakyReLU

    Parameters
    ----------
    channels: int
        The number of output channels (filters) in the convolution.
    kernel_size: int
        Specifies the dimension of the convolution window.
    strides: int
        Specify the strides of the convolution.
    padding: int
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.
    batch_norm: bool
        Use batch normalize after convolution or not.

    Returns
    -------
    mxnet.gluon.HybridSequential
    """
    out = nn.HybridSequential()
    with out.name_scope():
        out.add(nn.Conv2D(channels, kernel_size, strides, padding))
        if batch_norm:
            out.add(nn.BatchNorm())
        out.add(nn.LeakyReLU(0.1))
    return out


class ResidualLayer(HybridBlock):
    """Residual layer in yolov3.

    Parameters
    ----------
    channels: int
        Number of output channels (filters) in the convolutional layer.
    batch_norm: bool
        Whether to use batch normalize or not.
    """
    def __init__(self, channels, batch_norm=True):
        super(ResidualLayer, self).__init__()
        self.channels = channels
        self.batch_norm = batch_norm
        self.out = nn.HybridSequential()
        with self.name_scope():
            self.out.add(Convolutional(self.channels, 3, 1, 1, self.batch_norm))
            self.out.add(Convolutional(self.channels, 3, 1, 1, self. batch_norm))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return x+self.out(x)


def residual_block(channels, num_residual_layer):
    """Residual Block which stacks n residual layer.

    Parameters
    ----------
    channels : int
        Number of convolutional filters for every residual layer.
        Since residual layers in each block have the same number of channels, the type is int rather than list of int.
    num_residual_layer : int
        Number of residual layer in the residual block.
    """
    out = nn.HybridSequential()
    with out.name_scope():
        for i in range(num_residual_layer):
            out.add(ResidualLayer(channels))
        return out


class Darknet53(HybridBlock):
    """Darknet53 base network.

    Parameters
    ----------
    channels : list of int
        Number of convolution filters for each convolutional layer in darknet53 base network.
    num_residual_layer : list of int
        Number of Residual layer in each Residual Block.
    """
    def __init__(self, channels, num_residual_layer):
        super(Darknet53, self).__init__()
        self.channels = channels
        self.num_residual_layer = num_residual_layer
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.stages.add(Convolutional(self.channels[0], 3, 1, 1))
            for c, n in zip(self.channels[1:], self.num_residual_layer[1:]):
                self.stages.add(Convolutional(c, 3, 2, 1))
                self.stages.add(residual_block(c, n))

    def hybrid_forward(self, F, x, *args, **kwargs):
        # assert len(self.stages) == 11
        # out_1 = x
        # for stage in self.stages[:7]:
        #     out_1 = stage(out_1)
        # out_2 = out_1
        # for stage in self.stages[7:9]:
        #     out_2 = stage(out_2)
        # out_3 = out_2
        # for stage in self.stages[9:]:
        #     out_3 = stage(out_3)
        # return out_1, out_2, out_3

        assert len(self.stages) == 11
        outputs = []
        for stage in self.stages[:7]:
            x = stage(x)
        outputs.append(x)
        for stage in self.stages[7:9]:
            x = stage(x)
        outputs.append(x)
        for stage in self.stages[9:]:
            x = stage(x)
        outputs.append(x)
        return outputs

darknet_spec = {
    53: ([32, 64, 128, 256, 512, 1024], [0, 1, 2, 8, 8, 4])
}


def get_darknet53(num_layers):
    """Get Darknet53 feature extractor networks.

    Parameters
    ----------
    num_layers : int
        Darknet types, can be 53.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The returned networks.
    """
    channels, num_residual_layer = darknet_spec[num_layers]
    features = Darknet53(channels, num_residual_layer)
    return features


def darknet53_416():
    """Get Darknet53 feature extractor networks."""
    return get_darknet53(53)
