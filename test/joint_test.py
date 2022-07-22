import argparse
import torch
import torch.nn as nn
import time

from utils import *
from STG_model import STGModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_args():
    parser = argparse.ArgumentParser(description='AutoSTG')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--in_dim', type=int, default=1,
                        help='inputs dimension')
    parser.add_argument('--hid_dim', type=int, default=32,
                        help='for residual_channels and dilation_channels')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--num_nodes', type=int, default=170,
                        help='entities number')
    parser.add_argument('--randomadj', type=bool, default=True,  # random is better
                        help='whether random initialize adaptive adj')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)


def main(args):
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = True
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    adj_mx = get_adj_matrix('data/pems/PEMS08/PEMS08.csv', args.num_nodes)
    dataloader = generate_data('data/pems/PEMS08/PEMS08.npz', args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    model = STGModel(adj_mx, args)
    if args.cuda:
        model = model.cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters', params)

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    min_val_loss = float('inf')
    for epoch_num in range(args.epochs):
        print(f'epoch num: {epoch_num}')
        model = model.train()

        train_loss = []
        train_rmse = []
        dataloader['train_loader'].shuffle()
        t1 = time.time()
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(DEVICE)
            x = x.transpose(1, 3)
            y = torch.Tensor(y).to(DEVICE)  # [64, 12, 207, 2]
            y = y.transpose(1, 3)[:, 0, :, :]  # 二维标签只保留第一维，用于计算loss和mse


            optimizer.zero_grad()
            # x = nn.functional.pad(x, (1, 0, 0, 0))
            output = model(x)  # [64, 12, 207, 1]
            output = output.transpose(1, 3)
            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(output)

            # loss = huber_loss(predict, y, 0.0)
            loss = masked_mae(predict, y, 0.0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            train_loss.append(loss.item())
            rmse = masked_rmse(predict, y, 0.0)
            train_rmse.append(rmse.item())
        t2 = time.time()
        print(f"train: {t2-t1}")
        print(f'train_loss: {np.mean(train_loss)}, val_armse: {np.mean(train_rmse)}')

        # eval
        with torch.no_grad():
            model = model.eval()

            valid_loss = []
            valid_rmse = []
            t3 = time.time()
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(DEVICE)
                y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

                # x = nn.functional.pad(x, (1, 0, 0, 0))
                if i == 1:
                    t5 = time.time()
                output = model(x)
                if i == 1:
                    t6 = time.time()
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y = torch.unsqueeze(y, dim=1)
                predict = scaler.inverse_transform(output)

                # loss = huber_loss(predict, y, 0.0)
                loss = masked_mae(predict, y, 0.0)
                valid_loss.append(loss.item())
                rmse = masked_rmse(predict, y, 0.0)
                valid_rmse.append(rmse.item())

            t4 = time.time()
            print(f'infer1: {t4-t3}')
            print(f'infer2: {t6-t5}')
            val_loss = np.mean(valid_loss)
            print(f'val_loss: {val_loss}, val_armse: {np.mean(valid_rmse)}')

        if val_loss < min_val_loss:
            min_val_loss = val_loss

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
                    # y = torch.Tensor(y).to(DEVICE)
                    # y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

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
                    print(f'{i+1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])

                print(f'On average over 12 horizons, '
                      f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')


if __name__ == '__main__':
    args = build_args()
    main(args)
