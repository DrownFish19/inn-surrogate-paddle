import paddle.nn as nn


class Flat_data(nn.Layer):
    '''Args:
        X: input (BXCXHXW) 
        y: output (BXD)
        (This is used to flatten the data from 4D to 2D for the concat part of the fully connected layer).
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.reshape([x.shape[0], -1])
        return y
