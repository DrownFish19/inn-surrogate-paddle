from time import time

import h5py
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import scipy.io as io

from args import args
from models.conditioning_network import conditioning_network
from models.main_model import main_file
from utils.error_bars import error_bar, train_test_error
from utils.load_data import load_data

# load the data here
train_loader, test_loader, sample_loader, NLL_test_loader = load_data()
print('loaded the data.........', flush=True)


# this is the s and the t network

def convolution_network(Hidden_layer):
    return lambda input_channel, output_channel: nn.Sequential(
        nn.Conv3D(input_channel, Hidden_layer, (1, 3, 3), padding=(0, 1, 1)),
        nn.Dropout3D(p=0.1),
        nn.ReLU(),
        nn.Conv3D(Hidden_layer, output_channel, (1, 3, 3), padding=(0, 1, 1)))


def fully_connected(Hidden_layer):
    return lambda input_data, output_data: nn.Sequential(
        nn.Linear(input_data, Hidden_layer),
        nn.ReLU(),
        nn.Linear(Hidden_layer, output_data))


network_s_t = convolution_network(args.hidden_layer_channel)
network_s_t2 = convolution_network(args.hidden_layer_channel2)
network_s_t3 = fully_connected(args.hidden_layer3)
# load network
INN_network = main_file(args.cond_size, network_s_t,
                        args.input_dimension1, args.input_dimension12, args.cond_size1, args.permute_a1, args.split_channel,
                        args.input_dimension1_r,
                        args.input_dimension2, args.input_dimension22, args.cond_size2, args.permute_a2, network_s_t2,
                        args.input_dimension2_r,
                        args.input_dimension3, args.input_dimension32, args.cond_size3, network_s_t3, args.permute_a3)
cond_network = conditioning_network()

combine_parameters = [parameters_net for parameters_net in INN_network.parameters() if parameters_net.trainable]
for parameters_net in combine_parameters:
    parameters_net.set_value(0.02 * paddle.randn(parameters_net.shape))

combine_parameters += list(cond_network.parameters())
optimizer = optim.Adam(parameters=combine_parameters, learning_rate=args.lr, weight_decay=args.weight_decay)


def train():
    INN_network.train()
    cond_network.train()
    loss_mean = []
    loss_val = []
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.reshape([16, 1, 4, 64, 64])
        y = y.reshape([16, 4, 4, 64])  # for config_1  change this to y = y.reshape([16,2,4,64])
        y1 = cond_network(y)
        input = x
        c = y1[2]
        c2 = y1[1]
        c3 = y1[0]
        c4 = y1[3]
        z, log_j = INN_network(input, c, c2, c3, c4, forward=True)
        loss = paddle.mean(z ** 2) / 2 - paddle.mean(log_j) / (1 * 4 * 64 * 64)
        loss.backward()
        loss_mean.append(loss.item())
        optimizer.step()
        optimizer.clear_grad()
    loss_mean1 = loss_mean
    return loss_mean1


def test():
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    for batch_idx, (input, target) in enumerate(test_loader):
        x, y = input, target
        x = x.reshape([16, 1, 4, 64, 64])
        y = y.reshape([16, 4, 4, 64])  # for config_1  change this to y = y.reshape([16,2,4,64])
        input, target = x.reshape([16, 1, 4, 64, 64]), y.reshape([16, 4, 4, 64])  # for config_1  change this to target = y.reshape([16,2,
        # 4,64])
        x = input.reshape([16, 1, 4, 64, 64])
        y = target.reshape([16, 4, 4, 64])  # for config_1  change this to y = target.reshape([16,2,4,64])
        tic = time()
        y1 = cond_network(y)
        c = y1[2]
        c2 = y1[1]
        c3 = y1[0]
        c4 = y1[3]
        z, log_j = INN_network(x, c, c2, c3, c4, forward=True)
        loss_val = paddle.mean(z ** 2) / 2 - paddle.mean(log_j) / (1 * 4 * 64 * 64)
        loss_mean.append(loss_val.item())
    loss_mean1 = loss_mean
    return loss_mean1


def sample2(epoch):
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    for batch_idx, (input, target) in enumerate(sample_loader):
        x, y = input, target
        x = x.reshape([1, 1, 4, 64, 64])
        y = y.reshape([1, 4, 4, 64])  # for config_1  change this to y = y.reshape([1,2,4,64])
        input, target = x.reshape([1, 1, 4, 64, 64]), y.reshape([1, 4, 4, 64])  # for config_1  change this to target = y.reshape([1,2,4,
        # 64])
        x = input.reshape([1, 1, 4, 64, 64])
        y = target.reshape([1, 4, 4, 64])  # for config_1  change this to y = target.reshape([16,2,4,64])
        labels_test = target
        N_samples = 1000
        labels_test = labels_test[0, :, :, :]
        labels_test = labels_test.cpu().numpy()
        l = np.repeat(np.array(labels_test)[np.newaxis, :, :, :], N_samples, axis=0)
        l = paddle.Tensor(l)
        z = paddle.randn([N_samples, 16384])
        with paddle.no_grad():
            y1 = cond_network(l)
            input = x.reshape([1, 1, 4, 64, 64])
            c = y1[2]
            c2 = y1[1]
            c3 = y1[0]
            c4 = y1[3]
            val = INN_network(z, c, c2, c3, c4, forward=False)
        rev_x = val.cpu().numpy()
        if epoch == 200:
            input_test = input.cpu().numpy()
            f2 = h5py.File(f'{args.results_path}/data_save_{epoch}.h5', 'w')
            f2.create_dataset(f'{args.results_path}/input', data=input_test, compression='gzip', compression_opts=9)
            f2.create_dataset(f'{args.results_path}/pred', data=rev_x, compression='gzip', compression_opts=9)
            f2.close()
        if epoch % 20 == 0:
            input_test = input.cpu().numpy()
            input1 = input_test.reshape([1, 1, 4, 64, 64])
            samples1 = rev_x
            mean_samples1 = np.mean(samples1, axis=0)
            mean_samples1 = mean_samples1.reshape([1, 1, 4, 64, 64])  # mean of all the samples
            samples1 = samples1[:2, :, :, :, :]
            # =====
            mean_samples_layer1 = mean_samples1[0, 0, 1, :, :]
            mean_samples_layer1 = mean_samples_layer1.reshape([1, 1, 64, 64])
            input_layer1 = input1[0, 0, 1, :, :]
            input_layer1 = input_layer1.reshape([1, 1, 64, 64])
            samples_layer1 = samples1[:, 0, 1, :, :]
            samples_layer1 = samples_layer1.reshape([2, 1, 64, 64])
            x1 = np.concatenate((input_layer1, mean_samples_layer1, samples_layer1), axis=0)
            actual = input_layer1
            pred = rev_x[:, :, 1, :, :]
            pred = pred.reshape([1000, 1, 64, 64])
            error_bar(actual, pred, epoch, 1)


def test_NLL():
    domain = 16384
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    final_concat = []
    for batch_idx, (input, target) in enumerate(NLL_test_loader):
        x, y = input, target
        input, target = x.reshape([128, 1, 4, 64, 64]), y.reshape([128, 4, 4, 64])  # for config_1  change this to target = y.reshape([16,2,
        # 4,64])
        labels_test1 = target
        N_samples = 1000

        for jj in range(128):
            labels_test = labels_test1[jj, :, :, :]
            x = input[jj, :, :, :, :]
            labels_test = labels_test.cpu().numpy()
            l = np.repeat(np.array(labels_test)[np.newaxis, :, :, :], N_samples, axis=0)
            l = paddle.Tensor(l)
            z = paddle.randn([N_samples, 16384])
            with paddle.no_grad():
                y1 = cond_network(l)
                c = y1[2]
                c2 = y1[1]
                c3 = y1[0]
                c4 = y1[3]
                val = INN_network(z, c, c2, c3, c4, forward=False)
            rev_x = val.cpu().numpy()
            input1 = x.cpu().numpy()
            input1 = input1.reshape([1, 4, 64, 64])
            rev_x = rev_x.reshape([1000, 1, 4, 64, 64])
            mean_val = rev_x.mean(axis=0)
            mean_val = mean_val.reshape([1, 4, 64, 64])
            d1 = (1 / domain) * np.sum(input1 ** 2)
            n1 = (1 / domain) * np.sum((input1 - mean_val) ** 2)
            m1 = n1 / d1
            final_concat.append(m1)
        final_concat = np.array(final_concat)
    return final_concat


# for load model test
# model1 = f'/root/autodl-nas/3D/model-C2-5pc-10000/INN_network_epoch{epoch1}.pt'
# model2 = f'/root/autodl-nas/3D/model-C2-5pc-10000/cond_network_epoch{epoch1}.pt'
# print(model1, model2)
# INN_network.set_state_dict(paddle.load(model1))
# cond_network.set_state_dict(paddle.load(model2))

print('training start .............', flush=True)
loss_train_all = []
loss_test_all = []
tic = time()
for epoch in range(1, args.epochs):
    print('epoch number .......', epoch, flush=True)
    loss_train = train()
    loss_train2 = np.mean(loss_train)
    loss_train_all.append(loss_train2)
    with paddle.no_grad():
        sample2(epoch)
        loss_test = test()
        loss_test = np.mean(loss_test)
        print(('NLL loss:', loss_test), flush=True)
        loss_test_all.append(loss_test)

with paddle.no_grad():
    final_error = test_NLL()
    old_val = np.mean(final_error)
    print('relative l2:', np.mean(final_error), flush=True)

paddle.save(INN_network.state_dict(), f'{args.results_path}/INN_network_epoch{args.epochs}.pt')
paddle.save(cond_network.state_dict(), f'{args.results_path}/cond_network_epoch{args.epochs}.pt')
loss_train_all = np.array(loss_train_all)
loss_test_all = np.array(loss_test_all)
print('saving the training error and testing error', flush=True)
io.savemat(f'{args.results_path}/training_loss.mat', dict([('training_loss', np.array(loss_train_all))]))
io.savemat(f'{args.results_path}/test_loss.mat', dict([('testing_loss', np.array(loss_test_all))]))
print('plotting the training error and testing error', flush=True)
train_test_error(loss_train_all, loss_test_all, args.epochs)
toc = time()
print('total traning taken:', toc - tic, flush=True)
