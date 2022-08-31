import paddle.nn as nn


class Unflat_data(nn.Layer):
    """
    Args:
        X: input (BXD) to  output (BXCXHXW)
        This is used to unflatten the data from 2D to 4D for the concat part during the sampling phase
    """

    def __init__(self, input_dimension):
        super().__init__()
        self.shape_dim = input_dimension[0]

    def forward(self, x, sample_the_data=False):
        return x.reshape([x.shape[0], *self.shape_dim])
