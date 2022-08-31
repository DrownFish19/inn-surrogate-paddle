import paddle
import paddle.nn as nn


class CouplingOneSide(nn.Layer):
    """
    Args:
    s_t_network: scale and shit network
    input_dimension_1: Input dimension
    input_dimension_2: length of the input
    We use soft clamp as menioned in https://arxiv.org/abs/1907.02392 (Reference)

    """

    def __init__(self, s_t_network, condition_dimension):
        super().__init__()

        self.s_net = s_t_network(condition_dimension, 1)
        self.t_net = s_t_network(condition_dimension, 1)

    def jacobian(self):
        jacobian_val = self.jacobian_output
        return jacobian_val

    def forward(self, x, c, sample_the_data=False):
        x1, x2 = paddle.split(x, [0, 1], axis=1)
        if sample_the_data == False:
            x1_c = paddle.concat([x1, c], 1)
            self.s_network = self.s_net(x1_c)
            self.t_network = self.t_net(x1_c)
            y2 = (paddle.exp(1.1448 * paddle.atan(self.s_network))) * x2 + self.t_network
            output = paddle.concat((x1, y2), 1)
            jac = (1.1448 * paddle.atan(self.s_network))
            self.jacobian_output = paddle.sum(jac, axis=tuple(range(1, 4)))
            return output
        else:
            x1_c = paddle.concat([x1, c], 1)
            self.s_network = self.s_net(x1_c)
            self.t_network = self.t_net(x1_c)
            temp = (x2 - self.t_network) / (paddle.exp(1.1448 * paddle.atan(self.s_network)))
            output = paddle.concat((x1, temp), 1)
            jac = -(1.1448 * paddle.atan(self.s_network))
            self.jacobian_output = paddle.sum(jac, axis=tuple(range(1, 4)))
            return output
