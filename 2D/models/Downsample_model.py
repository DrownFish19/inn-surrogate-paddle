import paddle
import paddle.nn as nn


class Downsample(nn.Layer):
    """
    Args:
    Input: BXCXHXW
    Reference: Jacobsen et al.,"i-revnet: Deep invertible networks." for downsampling.
    """

    def __init__(self):
        super(Downsample, self).__init__()

    def forward(self, x, sample_the_data=False):
        if sample_the_data == True:

            batch_size, channel_1, height_1, width_1 = x.shape
            channel_2 = channel_1 / 4
            width_2 = width_1 * 2
            height_2 = (height_1) * 2
            data = x.transpose([0, 2, 3, 1])
            data_mod = data.reshape([batch_size, height_1, width_1, 4, int(channel_2)])
            val2 = []
            for data_s in data_mod.split(2, 3):
                val1 = data_s
                val1 = val1.reshape([1, batch_size, height_1, int(width_2), int(channel_2)])
                val2.append(val1)
            data1 = paddle.concat(val2, 0)
            data1 = data1.transpose([1, 0, 2, 3, 4])
            data = data1.transpose([0, 2, 1, 3, 4])
            data = data.reshape([batch_size, int(height_2), int(width_2), int(channel_2)])
            data = data.transpose([0, 3, 1, 2])
            return data
        else:
            batch_size, channel_2, height_2, width_2 = x.shape
            height_1 = height_2 / 2
            width_1 = width_2 / 2
            channel_1 = channel_2 * 4
            data = x.transpose([0, 2, 3, 1])
            val2 = []
            for data_s in data.split(int(data.shape[2] / 2), 2):
                val1 = data_s
                val1 = val1.reshape([int(batch_size), int(height_1), int(channel_1)])
                val2.append(val1)
            data2 = paddle.concat(val2, 1)
            data32 = data2.reshape([int(batch_size), int(height_1), int(width_1), int(channel_1)])
            data = data32.transpose([0, 2, 1, 3])
            data = data.transpose([0, 3, 1, 2])
            return data

# if __name__ == "__main__":
#     A = Downsample()
#     x = torch.Tensor(20,2,32,32)
#     B = A(x,sample_the_data = False)
#     print(B.shape)
