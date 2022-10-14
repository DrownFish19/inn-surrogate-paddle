import argparse
import os
import random
from pprint import pprint

import paddle

parser = argparse.ArgumentParser(description='Solving inverse problem using cINN')
# Block-1
parser.add_argument('--split_channel', type=list, default=[2, 2], help='Input dimension block-1 in C1,C2')
parser.add_argument('--cond_size', type=int, default=128, help='Conditioning dimension')
parser.add_argument('--hidden_layer_channel', type=int, default=64, help='number of channels for the hidden layer')
# Block-2
parser.add_argument('--input_dimension1_r', type=list, default=[(2, 32, 32)], help='Input dimension block-2 in CXHXW')
parser.add_argument('--cond_size1', type=int, default=128, help='Conditioning dimension')
parser.add_argument('--hidden_layer_channel1', type=int, default=64, help='number of channels for the hidden layer')
parser.add_argument('--input_dimension1', type=int, default=4, help='coupling block-2')
parser.add_argument('--input_dimension12', type=int, default=3, help='coupling block-2')
parser.add_argument('--permute_a1', type=int, default=4, help='permutation for the invertible block-2')
# Block-3
parser.add_argument('--input_dimension2_r', type=list, default=[(4, 16, 16)], help='Input dimension block-3 in CXHXW')
parser.add_argument('--cond_size2', type=int, default=48, help='Conditioning dimension')
parser.add_argument('--hidden_layer_channel2', type=int, default=96, help='number of channels for the hidden layer')
parser.add_argument('--input_dimension2', type=int, default=8, help='invertible block-3')
parser.add_argument('--input_dimension22', type=int, default=3, help='dinvertible block-3')
parser.add_argument('--permute_a2', type=int, default=8, help='permutation for the invertible block-3')
# training
parser.add_argument('--epochs', type=int, default=102, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0008, help='learnign rate')
parser.add_argument('--weight_decay', type=float, default=0.00005, help="weight decay")
parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing (default: 100)')
parser.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
parser.add_argument('--ntrain', type=int, default=10000, help="number of training data")
parser.add_argument('--ntest', type=int, default=128, help="number of test data")
# Block-4
parser.add_argument('--cond_size3', type=int, default=512, help='Conditioning dimension')
parser.add_argument('--hidden_layer3', type=int, default=4096, help='number of channels for the hidden layer')
parser.add_argument('--input_dimension3', type=int, default=1024, help='invertible block-4')
parser.add_argument('--input_dimension32', type=int, default=1, help='dinvertible block-4')
parser.add_argument('--permute_a3', type=int, default=1024, help='permutation for the invertible block-4')
# for run different datasets
parser.add_argument('--data_path', type=str, default='../data/2D_problem_dataset')  # 数据位置
parser.add_argument('--pc', type=int, default=1, help='[1,3,5]')  # 数据集噪声量，1%, 3%, 5%
parser.add_argument('--data_size', type=int, default=10000, help='[6000,8000,10000]')  # 训练集数量
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--results_path', type=str, default='results-2D')  # 结果保存位置

# 针对不同噪声数据，设置pc选项，data_size设置训练数据数量
args = parser.parse_args(args=['--pc', '1', '--data_size', '6000'])
# args = parser.parse_args(args=['--pc', '1', '--data_size', '8000'])
# args = parser.parse_args(args=['--pc', '1', '--data_size', '10000'])
# args = parser.parse_args(args=['--pc', '3', '--data_size', '6000'])
# args = parser.parse_args(args=['--pc', '3', '--data_size', '8000'])
# args = parser.parse_args(args=['--pc', '3', '--data_size', '10000'])
# args = parser.parse_args(args=['--pc', '5', '--data_size', '6000'])
# args = parser.parse_args(args=['--pc', '5', '--data_size', '8000'])
# args = parser.parse_args(args=['--pc', '5', '--data_size', '10000'])

# 输出超参数设置
# seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
print("Random Seed: ", args.seed, flush=True)
random.seed(args.seed)
paddle.seed(args.seed)
print('Arguments:', flush=True)
pprint(vars(args))
print(flush=True)

# global
device = paddle.set_device("gpu:{}".format(args.gpu_id) if paddle.is_compiled_with_cuda() else "cpu")

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
