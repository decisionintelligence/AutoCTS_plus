import argparse
import random
import math
import time
import numpy as np
import torch
import json
from scipy import stats
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pathlib import Path
from itertools import combinations
from functools import cmp_to_key

from joint_nac_engine2 import NAC, train_baseline_epoch, evaluate
from clean_set.genotypes import PRIMITIVES
# from utils import to_device2
from noisy_set.utils import joint_NAC_DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for zero-cost NAS')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--nac_lr', type=float, default=0.0001)
args = parser.parse_args()


def load_clean_set(root, epoch):
    # epoch1:100
    clean_set = []
    for i in range(1, 3):
        dir = root + '08/clean_joint3/' + f'clean{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)

        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            hyper = arch_pair['hyper']
            info = arch_pair['info'][:epoch+1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
                clean_set.append((arch, hyper, mae))

    random.shuffle(clean_set)
    return clean_set


def load_noisy_set(root):
    noisy_set = []

    # dir = root + '08/clean_joint3/' + f'noisy.json'
    # with open(dir, "r") as f:
    #     arch_pairs = json.load(f)
    # for arch_pair in arch_pairs:
    #     arch = arch_pair['arch']
    #     hyper = arch_pair['hyper']
    #     info = arch_pair['info'][0]
    #     mae = info[0]
    #     if mae < 120:
    #         noisy_set.append((arch, hyper, mae))

    for i in range(1, 6):
        dir = root + '08/clean_joint3/' + f'noisy{i}.json'
        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            hyper = arch_pair['hyper']
            info = arch_pair['info'][0]
            mae = info[0]
            if mae < 120:
                noisy_set.append((arch, hyper, mae))
        if i == 1:
            noisy_set = noisy_set[:400] + noisy_set[-400:]

    random.shuffle(noisy_set)
    return noisy_set


def load_pred_set(root):
    pred_set = []
    dir = root + '08/clean_joint3/' + f'pred.json'

    with open(dir, "r") as f:
        arch_pairs = json.load(f)

    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        hyper = arch_pair['hyper']
        pred_set.append((arch, hyper))

    random.shuffle(pred_set)
    return pred_set


def generate_pairs(data):
    # data: [(arch, hyper, mae)]
    pairs = []
    data = sorted(data, key=lambda x: x[-1])
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            pairs.append((data[i][0], data[i][1], data[j][0], data[j][1], 1))
            pairs.append((data[j][0], data[j][1], data[i][0], data[i][1], 0))

    return pairs


def main():

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    # 加载clean set和noisy set
    clean_data_dir = 'clean_set/'
    clean_set = load_clean_set(clean_data_dir, 99)
    noisy_set = load_noisy_set(clean_data_dir)
    # clean_set = sorted(clean_set, key=lambda x: x[-1])
    for pair in clean_set:
        print(pair)
    print(len(clean_set))
    print(len(noisy_set))

    # num = np.zeros([7, 7])
    # for i in [2, 4, 6]:
    #     for j in [4, 6]:
    #         for (arch, hyper, mae) in clean_set:
    #             if hyper[0] == i and hyper[1] == j:
    #                 num[i][j] += 1
    # print(num)
    #
    # num = np.zeros([7, 7])
    # for i in [2, 4, 6]:
    #     for j in [4, 6]:
    #         for (arch, hyper, mae) in noisy_set:
    #             if hyper[0] == i and hyper[1] == j:
    #                 num[i][j] += 1
    # print(num)

    # 分割clean数据集
    valid_set = []
    new_clean_set = []
    for i, (arch, hyper, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (arch2, hyper2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((arch, hyper, mae))
        else:
            new_clean_set.append((arch, hyper, mae))
    print(len(valid_set))
    print(len(new_clean_set))

    train_set = []
    remain_set = []
    for i, (arch, hyper, mae) in enumerate(new_clean_set):
        small_gap = 0
        for j, (arch2, hyper2, mae2) in enumerate(train_set):
            if abs(mae - mae2) < 0.01:
                small_gap = 1
                break
        if small_gap == 0:
            train_set.append((arch, hyper, mae))
        else:
            remain_set.append((arch, hyper, mae))
    print(len(train_set))
    print(len(remain_set))

    # num = np.zeros([7, 7])
    # for i in [2, 4, 6]:
    #     for j in [4, 6]:
    #         for (arch, hyper, mae) in valid_set:
    #             if hyper[0] == i and hyper[1] == j:
    #                 num[i][j] += 1
    # print(num)
    #
    # num = np.zeros([7, 7])
    # for i in [2, 4, 6]:
    #     for j in [4, 6]:
    #         for (arch, hyper, mae) in new_clean_set:
    #             if hyper[0] == i and hyper[1] == j:
    #                 num[i][j] += 1
    # print(num)

    valid_pairs = generate_pairs(valid_set)
    noisy_pairs = generate_pairs(noisy_set)
    remain_pairs = generate_pairs(new_clean_set)
    train_pairs = generate_pairs(train_set)
    # print(valid_pairs[:10])

    valid_loader = joint_NAC_DataLoader(valid_pairs, 1)
    noisy_loader = joint_NAC_DataLoader(noisy_pairs, args.batch_size)
    remain_loader = joint_NAC_DataLoader(remain_pairs, args.batch_size)
    train_loader = joint_NAC_DataLoader(train_pairs, args.batch_size)

    # build NAC
    nac = NAC(n_ops=len(PRIMITIVES), n_layers=4, embedding_dim=128).to(DEVICE)
    criterion = nn.BCELoss()
    nac_optimizer = optim.Adam(nac.parameters(), lr=args.nac_lr, betas=(0.5, 0.999), weight_decay=5e-4)

    his_loss = 100.
    tolerance = 0
    for epoch in range(100):  # 训练NAC多少个epochs？好像只能凭经验，因为没有test set
        valid_acc, loss = train_baseline_epoch(epoch,
                                               noisy_loader,
                                               valid_loader,
                                               nac,
                                               criterion,
                                               nac_optimizer)
        if loss < his_loss:
            tolerance = 0
            his_loss = loss
            torch.save(nac.state_dict(), "./saved_model/nac2" + ".pth")
        else:
            tolerance += 1
        if tolerance >= 5:
            break

    print('===================================================')
    nac.load_state_dict(torch.load("./saved_model/nac2" + ".pth"))
    his_loss = 100.
    tolerance = 0
    for epoch in range(100):
        valid_acc, loss = train_baseline_epoch(epoch,
                                               train_loader,
                                               valid_loader,
                                               nac,
                                               criterion,
                                               nac_optimizer)
        if loss < his_loss:
            tolerance = 0
            his_loss = loss
            torch.save(nac.state_dict(), "./saved_model/nac2" + ".pth")
        else:
            tolerance += 1
        if tolerance >= 5:
            break

    print('===================================================')
    nac.load_state_dict(torch.load("./saved_model/nac2" + ".pth"))

    # random sample and eval



if __name__ == '__main__':
    main()
