import h5py
import paddle
from paddle.io import DataLoader, TensorDataset

ntrain = 10000


def load_data():
    Train_hdf5_file = '../data/2D_problem_dataset/Config_2_train_obs_1pc.hdf5'
    with h5py.File(Train_hdf5_file, 'r') as f:
        x_train = f['input'][:ntrain]
        y_train = f['output'][:ntrain, :2, :]
        print('x_train:', x_train.shape, flush=True)
        print('y_train:', y_train.shape, flush=True)
        train_loader = DataLoader(TensorDataset([paddle.to_tensor(x_train, dtype='float32'), paddle.to_tensor(y_train, dtype='float32')]),
                                  batch_size=16, shuffle=True, drop_last=True)

    Test_hdf5_file = '../data/2D_problem_dataset/Config_2_test_obs_1pc.hdf5'
    with h5py.File(Test_hdf5_file, 'r') as f1:
        x_test = f1['input'][:, :]
        y_test_new = f1['output'][:, :2, :]
        print('x_test:', x_test.shape, flush=True)
        print('y_test:', y_test_new.shape, flush=True)
        test_loader = DataLoader(TensorDataset([paddle.to_tensor(x_test, dtype='float32'), paddle.to_tensor(y_test_new, dtype='float32')]),
                                 batch_size=16, shuffle=False, drop_last=True)
        test_loader_nll = DataLoader(
            TensorDataset([paddle.to_tensor(x_test, dtype='float32'), paddle.to_tensor(y_test_new, dtype='float32')]), batch_size=128,
            shuffle=False, drop_last=True)

    Sample_hdf5_file = '../data/2D_problem_dataset/Config_2_sample_obs_1pc.hdf5'
    with h5py.File(Sample_hdf5_file, 'r') as f2:
        x_test = f2['input'][:, :]
        y_test_new = f2['output'][:, :2, :]
        print('x_sample:', x_test.shape, flush=True)
        print('y_sample:', y_test_new.shape, flush=True)
        sample_loader = DataLoader(
            TensorDataset([paddle.to_tensor(x_test, dtype='float32'), paddle.to_tensor(y_test_new, dtype='float32')]), batch_size=1,
            shuffle=False, drop_last=True)

    # To load config-1 the make the channels as 1
    # for the observations as (B,2,obs)
    # For the train data: y_train_new_config_1 = y_train[:,:2,:]
    # For the test data: y_test_new_config_1 = y_test_new[:,:2,:]
    return train_loader, test_loader, sample_loader, test_loader_nll
