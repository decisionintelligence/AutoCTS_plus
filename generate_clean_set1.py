# cell_out clean sample collection
# 从08 clean sample里面选top80？运行cell out结果。
import os
import argparse
import numpy as np
import torch
import time
import random
import json

from utils import generate_data, get_adj_matrix, load_dataset, load_adj, masked_mae, masked_mape, masked_rmse, metric
from genotypes import PRIMITIVES
from st_net1 import Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--benchmark', dest='benchmark', type=str, default='08')
# parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
# parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
parser.add_argument('--adj_mx', type=str, default='data/METR-LA/adj_mx.pkl',
                    help='location of the data')
parser.add_argument('--data', type=str, default='data/METR-LA',
                    help='location of the adjacency matrix')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=307)
parser.add_argument('--hid_dim', type=int, default=32,
                    help='for residual_channels and dilation_channels')
parser.add_argument('--randomadj', type=bool, default=True,
                    help='whether random initialize adaptive adj')
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--layers', type=int, default=4, help='number of cells')
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
# parser.add_argument('--lr_min', type=float, default=0.0, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
# parser.add_argument('--grad_clip', type=float, default=5,
#                     help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


def load_clean_set(root, epoch):
    # epoch1:100
    clean_set = []
    for i in range(1, 4):
        if i == 1:
            dir = root + '08/trial3/' + 'clean_trans.json'
        else:
            dir = root + '08/trial3/' + f'clean_trans{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch+1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
                clean_set.append((arch, mae))

    for filename in ["pred_clean1.json", "pred_clean2.json", "pred_clean4.json", "pred_clean5.json"]:
        dir = root + '08/trial3/' + filename

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
                clean_set.append((arch, mae))

    random.shuffle(clean_set)
    return clean_set


class Random_NAS:
    def __init__(self, dataloader, adj_mx, scaler, save_dir):
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.adj_mx = adj_mx
        self.scaler = scaler

    def run(self):
        new_clean_set = []
        clean_data_dir = ''
        clean_set = load_clean_set(clean_data_dir, 99)
        print(len(clean_set))
        clean_set = sorted(clean_set, key=lambda x: x[-1])
        clean_set = clean_set[:80]
        old_archs, _ = zip(*clean_set)
        # print(clean_set)

        # archs = [[(0, 1), (0, 4), (1, 4), (1, 2), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 5), (1, 4), (0, 4), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 3), (2, 4), (2, 3), (3, 2)], [(0, 3), (0, 4), (1, 4), (1, 4), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 3), (1, 4), (0, 1), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 4), (1, 4), (1, 1), (2, 3), (2, 4), (3, 2)], [(0, 4), (0, 2), (1, 3), (1, 4), (2, 4), (2, 4), (3, 2)], [(0, 2), (0, 3), (1, 4), (1, 4), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 2), (1, 3), (1, 2), (2, 4), (2, 4), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 3), (2, 3), (2, 4), (3, 2)], [(0, 2), (0, 4), (1, 4), (1, 5), (2, 3), (2, 4), (3, 3)], [(0, 3), (0, 4), (1, 5), (1, 2), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 3), (2, 4), (2, 3), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 5), (2, 3), (2, 4), (3, 2)], [(0, 2), (0, 3), (1, 4), (0, 1), (2, 4), (2, 3), (3, 4)], [(0, 2), (0, 4), (1, 3), (1, 4), (2, 4), (2, 3), (3, 4)], [(0, 4), (0, 2), (1, 3), (1, 4), (2, 4), (2, 3), (3, 2)], [(0, 2), (0, 5), (1, 4), (0, 4), (2, 4), (2, 3), (3, 4)], [(0, 4), (0, 2), (1, 3), (1, 4), (2, 4), (2, 3), (3, 2)], [(0, 2), (0, 4), (1, 4), (0, 1), (2, 3), (2, 4), (3, 1)], [(0, 2), (0, 3), (1, 4), (0, 3), (2, 3), (2, 4), (3, 3)], [(0, 5), (0, 5), (1, 4), (1, 3), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 3), (2, 3), (2, 4), (3, 1)], [(0, 2), (0, 3), (1, 4), (0, 3), (2, 3), (2, 4), (3, 3)], [(0, 3), (0, 2), (1, 5), (1, 4), (2, 4), (2, 3), (3, 2)], [(0, 3), (0, 4), (1, 4), (1, 4), (2, 3), (2, 0), (3, 2)], [(0, 5), (0, 4), (1, 3), (1, 4), (2, 2), (2, 4), (3, 4)], [(0, 4), (0, 5), (1, 5), (1, 3), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 4), (1, 4), (1, 4), (2, 4), (2, 3), (3, 4)], [(0, 2), (0, 2), (1, 4), (0, 5), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 4), (1, 4), (1, 2), (2, 4), (2, 3), (3, 4)], [(0, 3), (0, 5), (1, 3), (1, 4), (2, 2), (2, 4), (3, 4)], [(0, 4), (0, 1), (1, 4), (1, 2), (2, 4), (2, 3), (3, 4)], [(0, 4), (0, 3), (1, 3), (0, 0), (2, 5), (2, 2), (3, 4)], [(0, 4), (0, 5), (1, 4), (0, 2), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 3), (1, 4), (0, 1), (2, 3), (2, 4), (3, 3)], [(0, 2), (0, 4), (1, 4), (0, 2), (2, 3), (2, 4), (3, 5)], [(0, 2), (0, 5), (1, 4), (0, 4), (2, 3), (2, 4), (3, 1)], [(0, 2), (0, 5), (1, 4), (0, 3), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 1), (1, 4), (0, 3), (2, 3), (2, 3), (3, 4)], [(0, 4), (0, 2), (1, 5), (1, 3), (2, 2), (2, 3), (3, 4)], [(0, 2), (0, 5), (1, 4), (0, 1), (2, 4), (2, 3), (3, 4)], [(0, 2), (0, 2), (1, 4), (0, 5), (2, 3), (2, 4), (3, 3)], [(0, 2), (0, 4), (1, 4), (1, 2), (2, 3), (2, 4), (3, 2)], [(0, 4), (0, 2), (1, 5), (1, 3), (2, 2), (2, 3), (3, 4)], [(0, 3), (0, 5), (1, 4), (1, 4), (2, 3), (2, 2), (3, 2)], [(0, 4), (0, 4), (1, 3), (1, 4), (2, 3), (2, 0), (3, 2)], [(0, 2), (0, 4), (1, 4), (1, 2), (2, 3), (2, 4), (3, 2)], [(0, 2), (0, 5), (1, 4), (0, 5), (2, 2), (2, 3), (3, 4)]]
        # archs = [i for n, i in enumerate(archs) if i not in archs[:n]]  # 去重

        # archs = [[(0, 2), (0, 4), (1, 4), (0, 3), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 3), (1, 3), (1, 2), (2, 4), (2, 4), (3, 2)], [(0, 4), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)], [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 3), (1, 3), (1, 4), (2, 4), (2, 4), (3, 2)], [(0, 2), (0, 4), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 4), (1, 4), (1, 3), (2, 3), (2, 4), (3, 2)], [(0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (3, 4)], [(0, 3), (0, 2), (1, 2), (1, 5), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 2), (1, 4), (1, 3), (2, 3), (2, 4), (3, 2)], [(0, 4), (0, 2), (1, 3), (0, 3), (2, 2), (1, 4), (3, 4)], [(0, 2), (0, 3), (1, 4), (1, 2), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 3), (1, 2), (1, 3), (2, 2), (0, 1), (3, 4)], [(0, 1), (0, 3), (1, 3), (1, 2), (2, 4), (2, 4), (3, 2)], [(0, 4), (0, 3), (1, 3), (0, 3), (2, 2), (1, 4), (3, 4)], [(0, 4), (0, 2), (1, 3), (1, 2), (2, 3), (2, 4), (3, 4)], [(0, 1), (0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)], [(0, 1), (0, 3), (1, 3), (1, 4), (2, 2), (0, 4), (3, 2)], [(0, 4), (0, 4), (1, 1), (1, 1), (2, 3), (2, 4), (3, 2)], [(0, 4), (0, 3), (1, 1), (1, 1), (2, 3), (2, 4), (3, 2)], [(0, 3), (0, 4), (1, 1), (1, 4), (2, 3), (2, 4), (3, 2)], [(0, 2), (0, 3), (1, 4), (0, 2), (2, 3), (2, 4), (3, 4)], [(0, 1), (0, 4), (1, 2), (0, 3), (2, 3), (2, 4), (3, 4)], [(0, 3), (0, 4), (1, 4), (1, 4), (2, 3), (2, 2), (3, 4)], [(0, 3), (0, 1), (1, 2), (1, 3), (2, 2), (0, 4), (3, 4)], [(0, 4), (0, 3), (1, 3), (0, 4), (2, 4), (2, 4), (3, 2)], [(0, 2), (0, 3), (1, 4), (0, 0), (2, 3), (2, 4), (3, 4)], [(0, 3), (0, 4), (1, 2), (0, 4), (2, 3), (2, 4), (3, 4)], [(0, 1), (0, 4), (1, 1), (1, 2), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 4), (1, 2), (1, 1), (2, 3), (0, 3), (3, 4)], [(0, 4), (0, 4), (1, 3), (0, 3), (2, 2), (1, 4), (3, 4)], [(0, 3), (0, 4), (1, 4), (1, 4), (2, 2), (0, 4), (3, 3)], [(0, 4), (0, 3), (1, 3), (1, 3), (2, 2), (1, 4), (3, 2)], [(0, 4), (0, 3), (1, 3), (1, 4), (2, 2), (1, 1), (3, 2)], [(0, 1), (0, 2), (1, 1), (0, 3), (2, 3), (2, 4), (3, 4)], [(0, 5), (0, 4), (1, 4), (0, 3), (2, 3), (2, 4), (3, 2)], [(0, 1), (0, 3), (1, 3), (0, 4), (2, 2), (1, 4), (3, 4)], [(0, 4), (0, 3), (1, 2), (0, 1), (2, 2), (1, 3), (3, 4)], [(0, 4), (0, 3), (1, 3), (1, 2), (2, 4), (0, 4), (3, 2)], [(0, 3), (0, 3), (1, 3), (1, 4), (2, 2), (1, 4), (3, 2)], [(0, 1), (0, 3), (1, 3), (0, 4), (2, 2), (1, 4), (3, 4)], [(0, 5), (0, 4), (1, 4), (1, 3), (2, 3), (2, 4), (3, 2)], [(0, 5), (0, 4), (1, 4), (1, 3), (2, 3), (2, 4), (3, 2)], [(0, 3), (0, 4), (1, 1), (1, 2), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 3), (1, 3), (0, 1), (2, 2), (1, 1), (3, 4)], [(0, 2), (0, 4), (1, 4), (0, 4), (2, 3), (2, 4), (3, 4)], [(0, 5), (0, 5), (1, 3), (1, 2), (2, 3), (0, 4), (3, 4)], [(0, 3), (0, 2), (1, 2), (0, 5), (2, 3), (2, 4), (3, 4)], [(0, 3), (0, 4), (1, 2), (0, 5), (2, 3), (2, 4), (3, 4)], [(0, 4), (0, 3), (1, 3), (1, 2), (2, 4), (1, 4), (3, 2)]]
        # archs = [i for n, i in enumerate(archs) if i not in archs[:n]]  # 去重

        archs = np.load('../noisy2000.npy')
        archs = archs.tolist()
        archs = [i for n, i in enumerate(archs) if i not in archs[:n]]  # 去重
        print(len(archs))

        for i in range(len(archs)):
            print(f'arch number: {i}')
            arch = archs[i]
            print(arch)

            if arch in old_archs:
                print('this arch exists in old clean set')
                continue

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # torch.backends.cudnn.benchmark = True
            if args.cuda:
                torch.backends.cudnn.deterministic = True

            # 返回每个epoch的metrics，100*3
            t1 = time.time()
            info = train_arch_from_scratch(self.dataloader, self.adj_mx, self.scaler, arch)
            print(f'arch{i} time: {time.time() - t1}')
            new_clean_set.append({"arch": np.array(arch).tolist(), "info": np.array(info).tolist()})
            with open(self.save_dir + '/cellout_noisy04_1.json', "w") as fw:
                json.dump(new_clean_set, fw)


def main():
    # Fill in with root output path
    root_dir = ''
    save_dir = os.path.join(root_dir, '%s/trial%d' % (args.benchmark, args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    # load dataset
    adj_mx = get_adj_matrix('../data/pems/PEMS04/PEMS04.csv', args.num_nodes)
    dataloader = generate_data('../data/pems/PEMS04/PEMS04.npz', args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    searcher = Random_NAS(dataloader, adj_mx, scaler, save_dir)
    searcher.run()


def train_arch_from_scratch(dataloader, adj_mx, scaler, arch):
    model = Network(adj_mx, scaler, args, arch)
    if args.cuda:
        model = model.cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters', params)

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    valid_metrics_list = []
    for epoch_num in range(args.epochs):
        print(f'epoch num: {epoch_num}')
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
        with torch.no_grad():  # 要统计三个metric，best of top n epochs，需要多个随机种子吗？加transformer或者Autoformer？
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
            print((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))

        # test
        if np.mean(valid_rmse) < 50.3:
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
                for i in range(12):
                    pred = scaler.inverse_transform(y_p[:, :, i])
                    real = y_t[:, :, i]
                    metrics = metric(pred, real)
                    print(f'{i + 1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])

                print(f'On average over 12 horizons, '
                      f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')

    return valid_metrics_list


if __name__ == '__main__':
    main()
