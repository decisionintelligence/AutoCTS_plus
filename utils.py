import pickle
import csv
import os
import pandas as pd
from pathlib import Path
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import numpy as np
import torch
from torch.autograd import Variable


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

# def generate_data(graph_signal_matrix_name, batch_size, test_batch_size=None, transformer=None):
#     """shape=[num_of_samples, 12, num_of_vertices, 1]"""
#
#     origin_data = np.load(graph_signal_matrix_name)  # shape=[17856, 170, 3]
#     keys = origin_data.keys()
#     if 'train' in keys and 'val' in keys and 'test' in keys:
#         data = generate_from_train_val_test(origin_data, transformer)
#
#     elif 'data' in keys:
#         length = origin_data['data'].shape[0]
#         data = generate_from_data(origin_data, length, transformer)
#
#     else:
#         raise KeyError("neither data nor train, val, test is in the data")
#
#     scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])
#
#     train_len = len(data['x_train'])
#
#     permutation = np.random.permutation(train_len)
#     data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
#     data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
#     data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
#     data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
#     data['x_train_3'] = copy.deepcopy(data['x_train_2'])
#     data['y_train_3'] = copy.deepcopy(data['y_train_2'])
#     data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
#     data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
#     data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)
#
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
#     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
#     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候
#
#     return data

def _generate_input_data(df, input_window=12, output_window=12):
    # df [17856, 170, 9]
    num_samples = df.shape[0]
    x_offsets = np.sort(np.concatenate((np.arange(-input_window + 1, 1, 1),)))
    y_offsets = np.sort(np.arange(1, output_window + 1, 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = df[t + x_offsets, ...]
        y_t = df[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def _split_train_val_test(x, y, train_rate=0.6, eval_rate=0.2):
    # print(x.shape)  # (17833, 12, 170, 9)
    test_rate = 1 - train_rate - eval_rate
    num_samples = x.shape[0]
    num_test = round(num_samples * test_rate)
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train

    part_train_rate = 1
    x_train, y_train = x[int(num_train*(1 - part_train_rate)):num_train, ...], y[int(num_train*(1 - part_train_rate)):num_train, ...]
    x_val, y_val = x[num_train: num_train + num_val, ...], y[num_train: num_train + num_val, ...]
    x_test, y_test = x[-num_test:, ...], y[-num_test:, ...]

    return x_train, y_train, x_val, y_val, x_test, y_test


def generate_data(graph_signal_matrix_name, batch_size, test_batch_size=None, adj_mx=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""

    origin_data = np.load(graph_signal_matrix_name)
    # print(origin_data['data'].shape)  # (17856, 170, 3)
    keys = origin_data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        data = generate_from_train_val_test(origin_data)

    elif 'data' in keys:
        length = origin_data['data'].shape[0]
        # print(origin_data['data'].shape)  # 在这里将(17856, 170, 3)转成(17856, 170, 9)
        origin_data = origin_data['data'][..., 0:1]
        num_samples, num_nodes, feature_dim = origin_data.shape
        data_list = [origin_data]

        # dynafile = pd.read_csv('data/pems/PEMS04/PeMS04.dyna')
        # timesolts = list(dynafile['time'][:int(dynafile.shape[0] / num_nodes)])
        # if not dynafile['time'].isna().any():
        #     timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        #     timesolts = np.array(timesolts, dtype='datetime64[ns]')

        # # time in day
        # time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")  # 0到1之间等间距划分上面的timesolts
        # time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))  # (17856, 170, 1) 每个节点，每个时间步都附上一个时间特征，0到1的float值
        # data_list.append(time_in_day)
        #
        # # day in week
        # dayofweek = []
        # for day in timesolts.astype("datetime64[D]"):
        #     # 根据timesolts判断该时间戳处于周几，比如周五对应数值4
        #     dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
        # day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))  # (17856, 170, 7)
        # day_in_week[np.arange(num_samples), :, dayofweek] = 1  # one-hot特征？
        # data_list.append(day_in_week)

        # # generate DTW matrix
        # time_intervals = 300  # 5 mins
        # points_per_hour = 3600 // time_intervals
        # # print('%' * 100)
        # # print(origin_data.shape)
        # # print(origin_data)
        # data_mean = np.mean(
        #     [origin_data[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
        #      for i in range(origin_data.shape[0] // (24 * points_per_hour))], axis=0)  # 每一天流量的平均值？
        # # print(data_mean.shape)
        # # print('=' * 100)
        # # print(data_mean)
        # dtw_distance = np.zeros((num_nodes, num_nodes))
        # for i in tqdm(range(num_nodes)):  # 在08数据集上需要几十分钟？
        #     for j in range(i, num_nodes):  # 比较不同节点的data_mean？感觉很粗糙啊？
        #         dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
        # for i in range(num_nodes):
        #     for j in range(i):
        #         dtw_distance[i][j] = dtw_distance[j][i]
        cache_path = "../data/pems/PEMS08/dtw_PeMS08.npy"
        # np.save(cache_path, dtw_distance)
        dtw_matrix = np.load(cache_path)

        # generate geo and sem masks
        sh_mx = adj_mx.copy()
        sh_mx[sh_mx > 0] = 1
        sh_mx[sh_mx == 0] = 511  # 这啥意思啊？
        for i in range(num_nodes):
            sh_mx[i, i] = 0
        for k in range(num_nodes):  # 统计两个节点之间的直接距离和间接距离？取最小值？
            for i in range(num_nodes):
                for j in range(num_nodes):
                    sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)

        sh_mx = sh_mx.T  # (170, 170) 为什么转置啊？
        far_mask_delta = 7  # mask稀疏程度
        dtw_delta = 5  # mask稀疏程度
        geo_mask = torch.zeros(num_nodes, num_nodes)
        geo_mask[sh_mx >= far_mask_delta] = 1
        geo_mask = geo_mask.bool()
        sem_mask = torch.ones(num_nodes, num_nodes)
        sem_mask_sort = dtw_matrix.argsort(axis=1)[:, :dtw_delta]
        for i in range(sem_mask.shape[0]):
            sem_mask[i][sem_mask_sort[i]] = 0
        sem_mask = sem_mask.bool()

        # data split
        data = np.concatenate(data_list, axis=-1)
        # data = generate_from_data(data, length)
        x, y = _generate_input_data(data, input_window=12, output_window=12)  # PDFormer data processing
        data = {}
        data['x_train'], data['y_train'], data['x_val'], data['y_val'], data['x_test'], data['y_test'] = \
            _split_train_val_test(x, y, train_rate=0.6, eval_rate=0.2)
    else:
        raise KeyError("neither data nor train, val, test is in the data")

    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar

    return data, geo_mask, sem_mask


def generate_from_train_val_test(origin_data):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')

    return data


def generate_from_data(origin_data, length):
    """origin_data shape: [17856, 170, 3]"""
    data = {}
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):
        x, y = generate_seq(origin_data['data'][line1: line2], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')

    return data


# def generate_seq(data, train_length, pred_length):
#     seq = np.concatenate([np.expand_dims(
#         data[i: i + train_length + pred_length], 0)
#         for i in range(data.shape[0] - train_length - pred_length + 1)],
#         axis=0)[:, :, :, 0: 1]
#     return np.split(seq, 2, axis=1)
def generate_seq(data, train_length, pred_length):
    # split data to generate x and y
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)

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
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape  # number of samples and nodes
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

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
# nac dataset processing
######################################################################
class NAC_DataLoader(object):
    def __init__(self, arch_pairs, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        x0, x1, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x0) % batch_size)) % batch_size
            x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
            x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            x0 = np.concatenate([x0, x0_padding], axis=0)
            x1 = np.concatenate([x1, x1_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
        self.size = len(x0)
        self.num_batch = int(self.size // self.batch_size)
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x0, x1, y = self.x0[permutation], self.x1[permutation], self.y[permutation]
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x0_i = self.x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = self.x1[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                yield x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()

