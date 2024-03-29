import pickle
import csv
import os
from pathlib import Path
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import copy
import numpy as np
import torch
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# dataset processing
######################################################################
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data_len[-int(data_len * (val_ratio + test_ratio)): -int(data_len * test_ratio)]
    train_data = data[: -int(data_len * (val_ratio + test_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    data format for seq2seq task or seq to single value task.
    :param data: shape [B, ...]
    :param window: history length
    :param horizon: future length
    :param single:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:  # 预测一个值
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window + horizon - 1: index + window + horizon])
            index += 1
    else:  # 预测下一个序列
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window: index + window + horizon])
            index += 1
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')

    return X, Y


def load_dataset(data_dir, batch_size, test_batch_size=None):
    """
    generate dataset
    :param data_dir:
    :param batch_size:
    :param test_batch_size:
    :return:
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        # print(cat_data['x'].shape)
        data['x_' + category] = cat_data['x'].astype('float32')
        data['y_' + category] = cat_data['y'].astype('float32')
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

    # data = {}
    # if 'pollution' not in data_dir and 'weather' not in data_dir:  # 数据集已分割
    #     for category in ['train', 'val', 'test']:
    #         cat_data = np.load(Path().joinpath(data_dir, category + '.npz'))
    #         data['x_' + category] = cat_data['x']
    #         data['y_' + category] = cat_data['y']
    #
    #     # 只考虑特征的第一维？计算loss和mse也是只考虑y的第一维
    #     scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    #     # Data format
    #     for category in ['train', 'val', 'test']:  # norm?
    #         data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])
    #
    #     # train_len = len(data['x_train'])
    #     # permutation = np.random.permutation(train_len)
    #     # data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
    #     # data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
    #     # data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
    #     # data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
    #     # data['x_train_3'] = copy.deepcopy(data['x_train_2'])
    #     # data['y_train_3'] = copy.deepcopy(data['y_train_2'])
    #     # data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
    #     # data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
    #     # data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)
    #
    #     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    #     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    #     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    #     data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候
    #
    #     return data
    # else:  # 分割并生成数据集
    #     dataset = np.load(data_dir, allow_pickle=True)
    #     data_train, data_val, data_test = split_data_by_ratio(dataset, 0.1, 0.2)
    #     x_tr, y_tr = Add_Window_Horizon(data_train, 12, 12, False)
    #     x_tr_orig = x_tr.copy()
    #     x_val, y_val = Add_Window_Horizon(data_val, 12, 12, False)
    #     x_test, y_test = Add_Window_Horizon(data_test, 12, 12, False)
    #     data['x_train'] = x_tr
    #     data['y_train'] = y_tr
    #     data['x_val'] = x_val
    #     data['y_val'] = y_val
    #     data['x_test'] = x_test
    #     data['y_test'] = y_test
    #
    #     real_scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    #     # Data format
    #     for category in ['train', 'val', 'test']:
    #         for i in range(x_tr.shape[-1]):
    #             scaler = StandardScaler(mean=x_tr_orig[..., i].mean(), std=x_tr_orig[..., i].std())
    #             data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i])
    #         print('x_' + category, data['x_' + category].shape)
    #
    #     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    #     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    #     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    #     data['scaler'] = real_scaler
    #
    #     return data


def load_adj(pkl_filename):
    """
    为什么gw的邻接矩阵要做对称归一化，而dcrnn的不做？其实做了，在不同的地方，是为了执行双向随机游走算法。
    所以K-order GCN需要什么样的邻接矩阵？
    这个应该参考ASTGCN，原始邻接矩阵呢？参考dcrnn
    为什么ASTGCN不采用对称归一化的拉普拉斯矩阵？
    :param pkl_filename: adj_mx.pkl
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx.astype('float32')


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pkl_filename, ':', e)
        raise

    return pkl_data


######################################################################
# generating chebyshev polynomials
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i-1] - cheb_polynomials[i-2])

    return cheb_polynomials


######################################################################
# generating diffusion convolution adj
######################################################################
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


######################################################################
# metrics
######################################################################
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def huber_loss(preds, labels, null_val=np.nan):
    mae = masked_mae(preds, labels, null_val)
    mse = masked_mse(preds, labels, null_val)
    loss = torch.where(mae < 1, 0.5 * mse, mae - 0.5)
    return loss


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


######################################################################
# PEMS03 ~ PEMS08 dataset
######################################################################
def generate_data(graph_signal_matrix_name, batch_size, test_batch_size=None, transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""

    origin_data = np.load(graph_signal_matrix_name)  # shape=[17856, 170, 3]
    # print(origin_data['data'][:, :, 0].shape)
    # print(origin_data['data'][0, 0, :])
    keys = origin_data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        data = generate_from_train_val_test(origin_data, transformer)

    elif 'data' in keys:
        length = origin_data['data'].shape[0]
        data = generate_from_data(origin_data, length, transformer)

    else:
        raise KeyError("neither data nor train, val, test is in the data")

    # print(data['x_train'].shape)
    # print(data['x_val'].shape)
    # print(data['x_test'].shape)
    # print('='*30)
    # print(data['x_train'][..., 0].std())
    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    train_len = len(data['x_train'])

    # permutation = np.random.permutation(train_len)
    # data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
    # data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
    # data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
    # data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
    # data['x_train_3'] = copy.deepcopy(data['x_train_2'])
    # data['y_train_3'] = copy.deepcopy(data['y_train_2'])
    # data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
    # data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
    # data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)

    # print(data['x_train'][900:1000, 6, 30, :])
    # print(data['x_train'][500:600, 2, 160, :])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候

    return data


def generate_from_train_val_test(origin_data, transformer):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data


def generate_from_data(origin_data, length, transformer):
    """origin_data shape: [17856, 170, 3]"""
    data = {}
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):

        x, y = generate_seq(origin_data['data'][line1: line2], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)
    # print(data['x_train'].shape)
    return data


def generate_seq(data, train_length, pred_length):
    # split data to generate x and y
    # print(data.shape)
    # aa = np.expand_dims(data[0: 0 + train_length + pred_length], 0)
    # print(aa.shape)
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]

    return np.split(seq, 2, axis=1)


def get_adj_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':  # 啥意思啊，表里有的就置1？
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return A


######################################################################
# Load long-range dataset
######################################################################
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object):
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        # train and valid is the ratio of training set and validation set. test = 1 - train - valid
        self.P = window  # 168
        self.h = horizon  # 3和24
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        # print(self.rawdat.shape)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape  # number of samples and nodes
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)  # test_y原始值

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)  # 对test_y原始值计算方差
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        # print(tmp.shape)
        # print(self.rse)
        # print(self.rae)

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


######################################################################
# Load electricity dataset
######################################################################
class StandardScaler_elec():
    """
    Standard the input
    """
    def __init__(self, data):
        self.m = data.shape[1]
        self.scale = np.ones(self.m)
        for i in range(self.m):
            self.scale[i] = np.max(np.abs(data[:, i]))

    def transform(self, data):
        new_data = np.zeros(data.shape)
        for i in range(self.m):
            new_data[:, :, i, 0] = data[:, :, i, 0] / self.scale[i]

        return new_data

    def inverse_transform(self, data):
        new_data = np.zeros(data.shape)
        for i in range(self.m):
            new_data[:, :, i, 0] = data[:, :, i, 0] * self.scale[i]
        return new_data


def generate_from_data_elec(origin_data, length):
    """origin_data shape: [17856, 170, 1]"""
    data = {}
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):

        x, y = generate_seq(origin_data[line1: line2], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')

    return data


def generate_elec_data(file_name, batch_size, test_batch_size = None):
    fin = open(file_name)
    rawdat = np.loadtxt(fin, delimiter=',')

    # for i in range(rawdat.shape[1]):  # 不能做两次标准化
    #     rawdat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))
    scalar = StandardScaler_elec(rawdat)
    rawdat = np.expand_dims(rawdat, -1)  # (26304, 321, 1)
    length = rawdat.shape[0]

    # print(rawdat[-5:, 0:10, 0])
    data = generate_from_data_elec(rawdat, length)
    # print(data['x_train'].shape)
    # print(data['x_train'][:3, :, 1, :])
    # print(data['x_train'][..., 0].mean())
    # print(data['x_train'][..., 0].std())
    # scalar = StandardScaler_elec(rawdat)
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scalar.transform(data['x_' + category])

    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scalar.inverse_transform(data['x_' + category][..., 0])

    # print(data['x_train'].shape)
    # print(data['x_val'].shape)
    # print(data['y_test'].shape)
    # print(data['x_train'][500:600, 2, 100, :])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar

    return data


if __name__ == '__main__':
    Data = generate_elec_data('data/electricity.txt', 64, 64)
    # dataloader = generate_data('data/pems/PEMS08/PEMS08.npz', 64, 64)
    # Data = DataLoaderS('data/electricity.txt', 0.6, 0.2, DEVICE, 3, 168, 2)

