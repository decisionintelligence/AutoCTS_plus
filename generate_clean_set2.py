# generate a small number of (arch, acc) pairs for each hyper-parameter setting

import os
import argparse
import numpy as np
import torch
import time
import random
import json

from utils import generate_data, get_adj_matrix, load_dataset, load_adj, masked_mae, masked_mape, masked_rmse, metric
from genotypes import PRIMITIVES
from st_net2 import Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--benchmark', dest='benchmark', type=str, default='03')
# parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
# parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
parser.add_argument('--adj_mx', type=str, default='data/METR-LA/adj_mx.pkl',
                    help='location of the data')
parser.add_argument('--data', type=str, default='data/METR-LA',
                    help='location of the adjacency matrix')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=358)
parser.add_argument('--hid_dim', type=int, default=32,
                    help='for residual_channels and dilation_channels')
parser.add_argument('--out_dim', type=int, default=512,
                    help='for end_linear_2')
parser.add_argument('--dropout', type=bool, default=False,
                    help='False: dropout rate=0. True: dropout rate=0.3')
parser.add_argument('--cell_out', type=bool, default=False,
                    help='False: using the last node as the output of a cell. True: using 1st and 3rd instead.')
parser.add_argument('--randomadj', type=bool, default=True,
                    help='whether random initialize adaptive adj')
parser.add_argument('--seq_len', type=int, default=48)
parser.add_argument('--layers', type=int, default=4, help='number of cells')
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
# parser.add_argument('--lr_min', type=float, default=0.0, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
# parser.add_argument('--grad_clip', type=float, default=5,
#                     help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


class Random_NAS:
    def __init__(self, dataloader, adj_mx, scaler, save_dir):
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.adj_mx = adj_mx
        self.scaler = scaler

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if args.cuda:
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def sample_arch():
        # k = sum(1 for i in range(self.args.steps) for n in range(1 + i))  # 计算DAG的edge_num
        num_ops = len(PRIMITIVES)
        n_nodes = args.steps

        arch = []  # 要改，不能选none？再加个pad函数，补齐为6个节点
        for i in range(n_nodes):
            if i == 0:
                ops = np.random.choice(range(num_ops), 1)
                nodes = np.random.choice(range(i + 1), 1)
                arch.extend([(nodes[0], ops[0])])
            else:
                ops = np.random.choice(range(num_ops), 2)  # 两条input edge对应两个op（可以相同）
                nodes = np.random.choice(range(i), 1)  # 只有一条可以选择的边
                # nodes = np.random.choice(range(i + 1), 2, replace=False)
                arch.extend([(nodes[0], ops[0]), (i, ops[1])])

        return arch

    def run(self):
        # arch_hyper_set = []
        clean_set = []

        # layers = [6]  # 采用序号编码？
        # steps = [6]
        # hid_dim = [32]
        # out_dim = [256]
        # dropout = [True]
        # cell_out = [True]
        #
        #
        # t1 = time.time()
        # while len(arch_hyper_set) < 800:  # 应该放在循环里面？也不行，如果某一类由于参数量原因数量就是比较少呢？
        #     for l in layers:
        #         for s in steps:
        #             for h in hid_dim:
        #                 for o in out_dim:
        #                     for d in dropout:
        #                         for c in cell_out:
        #                             args.layers = l
        #                             args.steps = s
        #                             args.hid_dim = h
        #                             args.out_dim = o
        #                             args.dropout = d
        #                             args.cell_out = c
        #
        #                             # arch_hypers = []
        #                             # for i in range(1000):  # 每个setting采样1000个网络，选择其中符合条件的5个作为clean sample？
        #                             #     t2 = time.time()
        #                             #     arch = self.sample_arch()
        #                             #     indices, ops = zip(*arch)
        #                             #
        #                             #     if not (1 in ops and 2 in ops and 3 in ops):  # 要求同时包含dcc, gcn, trans
        #                             #         continue
        #                             #     if ops.count(0) > s - 2:  # skip操作不能大于等于3个或5个？
        #                             #         continue
        #                             #
        #                             #     model = Network(self.adj_mx, self.scaler, args, arch)
        #                             #     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #                             #     print(f'build time: {time.time() - t2}')
        #                             #     if 300000 < params < 400000:
        #                             #         arch_hypers.append((arch, l, s, h, o, d, c))
        #                             # if len(arch_hypers) > 10:
        #                             #     # random.shuffle(arch_hypers)
        #                             #     arch_hyper_set.extend(arch_hypers)
        #
        #                             t2 = time.time()
        #                             arch = self.sample_arch()
        #                             indices, ops = zip(*arch)
        #                             if not (1 in ops and 2 in ops and 3 in ops):  # 要求同时包含dcc, gcn, trans
        #                                 continue
        #                             if ops.count(0) > s - 2:  # skip操作不能大于等于3个或5个？去掉这个条件吧！
        #                                 continue
        #
        #                             model = Network(self.adj_mx, self.scaler, args, arch)
        #                             params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #                             # print(f'build time: {time.time() - t2}')
        #                             if 300000 < params < 400000:
        #                                 arch_hyper_set.append((arch, l, s, h, o, d, c))
        #
        # print(len(arch_hyper_set))
        # print(f'sample time: {time.time() - t1}')


        arch_hyper_set = [([[0, 1], [0, 1], [1, 0], [1, 1], [2, 2], [1, 4], [3, 1], [0, 3], [4, 0], [2, 3], [5, 0]], 6, 6, 32, 256, 1, 1)]

        for i in range(len(arch_hyper_set)):
            print(f'arch number: {i}')
            arch, l, s, h, o, d, c = arch_hyper_set[i]
            args.layers = l
            args.steps = s
            args.hid_dim = h
            args.out_dim = o
            args.dropout = d
            args.cell_out = c
            print(arch, l, s, h, o, d, c)

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            if args.cuda:
                torch.backends.cudnn.deterministic = True

            # 返回每个epoch的metrics，100*3
            t1 = time.time()
            # info = [0]
            info = train_arch_from_scratch(self.dataloader, self.adj_mx, self.scaler, arch)
            print(f'arch{i} time: {time.time() - t1}')
            clean_set.append({"arch": np.array(arch).tolist(),
                              "hyper": np.array([l, s, h, o, d, c]).tolist(),
                              "info": np.array(info).tolist()})
            with open(self.save_dir + '/clean4.json', "w") as fw:
                json.dump(clean_set, fw)


def main():
    # Fill in with root output path
    root_dir = ''
    save_dir = os.path.join(root_dir, '%s/clean_joint%d' % (args.benchmark, args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    # load dataset
    # _, _, adj_mx = load_adj(args.adj_mx)
    # dataloader = load_dataset(args.data, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']

    adj_mx = get_adj_matrix('../data/pems/PEMS03/PEMS03.csv', args.num_nodes, id_filename='../data/pems/PEMS03/PEMS03.txt')
    dataloader = generate_data('../data/pems/PEMS03/PEMS03.npz', args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    searcher = Random_NAS(dataloader, adj_mx, scaler, save_dir)
    searcher.run()


def train_arch_from_scratch(dataloader, adj_mx, scaler, arch):
    model = Network(adj_mx, scaler, args, arch)
    if args.cuda:
        model = model.cuda()

    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total number of parameters', params)

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    valid_metrics_list = []
    for epoch_num in range(args.epochs):
        # print(f'epoch num: {epoch_num}')
        model = model.train()

        dataloader['train_loader'].shuffle()
        t2 = time.time()
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(DEVICE)
            x = x.transpose(1, 3)
            y = torch.Tensor(y).to(DEVICE)  # [64, 12, 207, 2]
            y = y.transpose(1, 3)[:, 0, :, :]

            optimizer.zero_grad()
            output = model(x)  # [64, 12, 207, 1]
            output = output.transpose(1, 3)
            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(output)  # unnormed x

            loss = masked_mae(predict, y, 0.0)  # y也是unnormed
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f'train epoch time: {time.time() - t2}')

        # eval
        with torch.no_grad():  # 要统计三个metric，best of top n epochs，需要多个随机种子吗？
            model = model.eval()

            valid_loss = []
            valid_rmse = []
            valid_mape = []
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(DEVICE)
                y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y = torch.unsqueeze(y, dim=1)
                predict = scaler.inverse_transform(output)

                loss = masked_mae(predict, y, 0.0)
                rmse = masked_rmse(predict, y, 0.0)
                mape = masked_mape(predict, y, 0.0)
                valid_loss.append(loss.item())
                valid_rmse.append(rmse.item())
                valid_mape.append(mape.item())
            valid_metrics_list.append((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))

            # test
            if np.mean(valid_rmse) < 100.3:
                with torch.no_grad():
                    model = model.eval()

                    y_p = []
                    y_t = torch.Tensor(dataloader['y_test']).to(DEVICE)
                    y_t = y_t.transpose(1, 3)[:, 0, :, :]
                    for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                        x = torch.Tensor(x).to(DEVICE)
                        x = x.transpose(1, 3)

                        # x = nn.functional.pad(x, (1, 0, 0, 0))
                        output = model(x)
                        output = output.transpose(1, 3)  # [64, 1, 207, 12]
                        y_p.append(output.squeeze())

                    y_p = torch.cat(y_p, dim=0)
                    y_p = y_p[:y_t.size(0), ...]

                    amae = []
                    amape = []
                    armse = []
                    for i in range(48):
                        pred = scaler.inverse_transform(y_p[:, :, i])
                        real = y_t[:, :, i]
                        metrics = metric(pred, real)
                        # print(f'{i + 1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                        amae.append(metrics[0])
                        amape.append(metrics[1])
                        armse.append(metrics[2])

                    print(f'On average over 12 horizons, '
                          f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')

    return valid_metrics_list


if __name__ == '__main__':
    main()
