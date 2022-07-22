import numpy as np
import torch
import torch.nn as nn
from torch.optim.sgd import SGD

from gcn_net import GCN
from genotypes import PRIMITIVES
# from utils import to_device, shuffle
# from metric import AccuracyMetric, AverageMetric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_baseline_epoch(epoch, data_loader, valid_loader, nac, criterion, optimizer):
    train_loss = []
    num_correct = 0
    nac.train()
    train_dataloader = data_loader
    print(f'epoch num: {epoch}')
    train_dataloader.shuffle()

    for i, (arch0, hyper0, arch1, hyper1, label) in enumerate(train_dataloader.get_iterator()):  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)
        outputs = nac(arch0, hyper0, arch1, hyper1)

        loss = criterion(outputs, label)
        train_loss.append(loss.item())
        pred = torch.round(outputs)
        num_correct += torch.eq(pred, label).sum().float().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = num_correct / train_dataloader.size
    print(f'acc: {accuracy} train_loss: {np.mean(train_loss)}')
    print('Computing Test Result...')

    # eval
    with torch.no_grad():
        nac = nac.eval()
        valid_loss = []
        num_correct = 0
        for i, (arch0, hyper0, arch1, hyper1, label) in enumerate(valid_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)
            outputs = nac(arch0, hyper0, arch1, hyper1)

            pred = torch.round(outputs)
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())
            num_correct += torch.eq(pred, label).sum().float().item()

        accuracy = num_correct / valid_loader.size

    print(f'valid_acc: {accuracy}, valid_loss: {np.mean(valid_loss)}')

    return accuracy, np.mean(valid_loss)


def evaluate(val_loader, nac, criterion):
    with torch.no_grad():
        nac.eval()
        num_correct = 0
        valid_loss = []
        for i, (arch0, arch1, label) in enumerate(val_loader.get_iterator()):
            # arch0 = torch.Tensor(arch0).to(DEVICE)
            # arch1 = torch.Tensor(arch1).to(DEVICE)
            label = torch.Tensor(label).to(DEVICE)
            outputs = nac(arch0, arch1)
            loss = criterion(outputs, label)
            pred = torch.round(outputs)
            valid_loss.append(loss.item())
            num_correct += torch.eq(pred, label).sum().float().item()
        accuracy = num_correct / val_loader.size

    return accuracy, np.mean(valid_loss)


def geno_to_adj(arch):
    # arch.shape = [11, 2]
    # 输出邻接矩阵，和节点特征
    # GCN处理无向图，这里DAG是有向图，所以需要改改？？？参考Wei Wen的文章
    node_num = len(arch) + 2  # 加上一个input和一个output节点
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]  # input节点
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    ops.append(len(PRIMITIVES) + 1)

    pad_row = np.array([[1]*node_num])
    adj = np.insert(adj, 0, values=pad_row, axis=0)
    pad_col = np.array([[0]*(node_num + 1)])
    adj = np.insert(adj, 0, values=pad_col, axis=1)

    return adj, ops


def geno_to_adj_multi_out(arch):
    # arch.shape = [11, 2]
    # cell的输出方式和GW一样，基数节点作为输出
    # 好像不太好搞，暂时不搞吧
    node_num = len(arch) + 2  # 加上一个input和一个output节点
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]  # input节点
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    ops.append(len(PRIMITIVES) + 1)

    return adj, ops


def geno_to_adj_pad(arch):
    # arch.shape = [7, 2]
    # 只有四个内部节点，需要padding，使得邻接矩阵和6个内部节点的网络shape相同
    node_num = len(arch) + 2  # 加上一个input和一个output节点
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]  # input节点
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    # ops.append(len(PRIMITIVES) + 1)

    pad_row = np.array([[0]*node_num])
    for i in range(4):  # 11-7=4
        adj = np.insert(adj, node_num + i - 1, values=pad_row, axis=0)
    pad_col = np.array([[0]*(node_num + 4)])
    for i in range(4):
        adj = np.insert(adj, node_num + i - 1, values=pad_col, axis=1)
        ops.append(len(PRIMITIVES) + 2)  # 对应none节点，padding部分

    ops.append(len(PRIMITIVES) + 1)

    pad_row = np.array([[1] * (node_num + 4)])
    adj = np.insert(adj, 0, values=pad_row, axis=0)
    pad_col = np.array([[0] * (node_num + 5)])
    adj = np.insert(adj, 0, values=pad_col, axis=1)

    return adj, ops


def transform_hypers(hypers):
    # max-min normalization
    new_hypers = []
    new_hypers.append((hypers[0] - 2) / 4)
    new_hypers.append((hypers[1] - 4) / 2)
    new_hypers.append((hypers[2] - 32) / 32)
    new_hypers.append((hypers[3] - 128) / 384)
    new_hypers.append(hypers[4] / 1)
    new_hypers.append(hypers[5] / 1)

    return new_hypers


class NAC(nn.Module):
    def __init__(self, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        super(NAC, self).__init__()
        self.n_ops = n_ops

        # +2用于表示input和output node，+3多一个表示padding或者none操作？
        self.embedding = nn.Embedding(self.n_ops + 3, embedding_dim=embedding_dim).to(DEVICE)
        self.hyper_embedding = nn.Parameter(torch.randn(6, embedding_dim).to(DEVICE), requires_grad=True).to(DEVICE)
        self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim).to(DEVICE)

        self.fc = nn.Linear(embedding_dim * ratio, 1, bias=True).to(DEVICE)  # f_out=1  ratio是啥意思？

    def forward(self, arch0, hyper0, arch1, hyper1):
        # arch0.shape = [batch_size, 11, 2]

        b_adj0, b_adj1, b_ops0, b_ops1, b_hyper0, b_hyper1 = [], [], [], [], [], []
        for i in range(len(arch0)):
            if hyper0[i][1] == 4:
                adj0, ops0 = geno_to_adj_pad(arch0[i][:-4])
            else:
                adj0, ops0 = geno_to_adj(arch0[i])
            if hyper1[i][1] == 4:
                adj1, ops1 = geno_to_adj_pad(arch1[i][:-4])
            else:
                adj1, ops1 = geno_to_adj(arch1[i])

            b_adj0.append(adj0)
            b_adj1.append(adj1)
            b_ops0.append(ops0)
            b_ops1.append(ops1)
            b_hyper0.append(transform_hypers(hyper0[i]))
            b_hyper1.append(transform_hypers(hyper1[i]))

        b_adj0 = torch.Tensor(b_adj0).to(DEVICE)
        b_adj1 = torch.Tensor(b_adj1).to(DEVICE)
        b_ops0 = torch.LongTensor(b_ops0).to(DEVICE)
        b_ops1 = torch.LongTensor(b_ops1).to(DEVICE)
        b_hyper0 = torch.Tensor(b_hyper0).to(DEVICE)
        b_hyper1 = torch.Tensor(b_hyper1).to(DEVICE)
        feature = torch.cat([self.extract_features((b_adj0, b_ops0, b_hyper0)),
                             self.extract_features((b_adj1, b_ops1, b_hyper1))], dim=1)

        score = self.fc(feature).view(-1)

        probility = torch.sigmoid(score)

        return probility

    def extract_features(self, arch):
        # 分别输入邻接矩阵和operation？
        if len(arch) == 3:
            matrix, op, hyper = arch
            return self._extract(matrix, op, hyper)
        else:
            print('error')

    def _extract(self, matrix, ops, hyper):
        # 这里ops是序号编码
        hyper = torch.mm(hyper, self.hyper_embedding)
        hyper = hyper.unsqueeze(1)
        ops = self.embedding(ops)
        ops = torch.cat([hyper, ops], dim=1)  # 拼接hyper节点的embedding
        # feature = self.gcn(ops, matrix)[:, 0, :]  # hyper节点作为输出
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling

        return feature


if __name__ == '__main__':
    arch = [(0, 2), (0, 0), (1, 1), (1, 3), (2, 3), (2, 4), (3, 1)]
    adj, ops = geno_to_adj(arch)
    print(adj)
    print(ops)

    # arch = [(0, 2), (0, 0), (1, 1), (1, 3), (2, 3), (2, 4), (3, 1), (0, 0), (0, 0), (0, 0), (0, 0)]
    adj, ops = geno_to_adj_pad(arch)
    print(adj)
    print(ops)

    hyper = [6, 6, 64, 128, 0, 1]
    hyper = transform_hypers(hyper)
    print(hyper)

    # arch = [(0, 2), (0, 0), (1, 1), (1, 3), (2, 3), (2, 4), (3, 1), (3, 4), (4, 4), (4, 4), (5, 4)]
    # adj, ops = geno_to_adj(arch)
    # print(adj)
    # print(ops)
