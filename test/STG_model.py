import torch
import torch.nn as nn

from building_blocks import *
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STGModel(nn.Module):
    def __init__(self, adj_mx, args, kernel_size=2):
        super(STGModel, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.seq_len
        self.layer_num = 4

        self.dccs = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.skip_connect = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.pe = PositionalEmbedding(self.hid_dim)

        # modeling spatial dependency with K-order gcn layer
        new_adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(DEVICE) for i in new_adj]

        # adaptive adjacency matrix
        # self.alpha = nn.Parameter(torch.randn(len(supports), len(supports)).to(DEVICE), requires_grad=True).to(DEVICE)
        if args.randomadj:
            aptinit = None
        else:
            aptinit = supports[0]
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(args.num_nodes, 10).to(DEVICE), requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(torch.randn(10, args.num_nodes).to(DEVICE), requires_grad=True).to(DEVICE)
            # self.nodevec1 = nn.Parameter(torch.randn(12, args.num_nodes, 10).to(DEVICE), requires_grad=True).to(DEVICE)
            # self.nodevec2 = nn.Parameter(torch.randn(12, 10, args.num_nodes).to(DEVICE), requires_grad=True).to(DEVICE)
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(DEVICE)

        for i in range(self.layer_num * 2):
            # if i % 2 == 0:
            #     self.dccs.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=1))
            # else:
            #     self.dccs.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=2))
            #
            # # self.gcns.append(GCNLayer(2, cheb_polynomials, c_in=self.hid_dim, c_out=self.hid_dim))
            #
            # self.gcns.append(DiffusionConvLayer(2, supports, self.hid_dim, self.hid_dim, self.nodevec1, self.nodevec2))

            self.skip_connect.append(nn.Conv1d(self.hid_dim, self.hid_dim * 8, kernel_size=(1, 1)))

            # self.bn.append(nn.BatchNorm2d(self.hid_dim))

        self.dcc1 = nn.ModuleList()
        self.dcc2 = nn.ModuleList()
        self.diff_gcn = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.s_trans = nn.ModuleList()
        self.preprocess = nn.ModuleList()

        for i in range(4):
            # self.s_trans.append(SpatialInformer(self.hid_dim))
            # self.trans.append((InformerLayer(self.hid_dim)))
            # self.dcc1.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=1))
            # self.dcc2.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=2))
            self.diff_gcn.append(
                DiffusionConvLayer(2, supports, self.hid_dim, self.hid_dim, self.nodevec1, self.nodevec2))

        for i in range(8):
            # self.trans.append((InformerLayer(self.hid_dim)))
            # self.s_trans.append(SpatialInformer(self.hid_dim))
            self.dcc1.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=1))
            # self.dcc2.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=2))
            # self.diff_gcn.append(
            #     DiffusionConvLayer(2, supports, self.hid_dim, self.hid_dim, self.nodevec1, self.nodevec2))

        for i in range(12):
            self.trans.append((InformerLayer(self.hid_dim)))
            # self.dcc2.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=2))
            # self.dcc1.append(DCCLayer(self.hid_dim, self.hid_dim, (1, kernel_size), dilation=1))
            # self.diff_gcn.append(
            #     DiffusionConvLayer(2, supports, self.hid_dim, self.hid_dim, self.nodevec1, self.nodevec2))

        self.start_linear = linear(self.in_dim, self.hid_dim)

        # output layer
        self.end_linear_1 = linear(self.hid_dim * 8, self.hid_dim * 16)
        self.end_linear_2 = linear(self.hid_dim * 16, self.out_dim)

    def cell(self, inputs, dcc1_1, dcc1_2, trans1, trans2, trans3, diff):
        x0 = inputs
        x1 = x0
        x2 = trans1(x0) + trans2(x1)
        x3 = dcc1_1(x1) + diff(x2)
        x4 = trans3(x2) + dcc1_2(x3)

        return x1, x3, x4

    def forward(self, inputs):
        """
        :param inputs: [bsize, in_dim, num_nodes, seq_len]
        :return:
        """
        x = inputs
        x = self.start_linear(x)  # [64, 32, 207, 12]
        skip = 0

        b, D, N, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, T, D)  # [64, 207, 12, 32]
        x = x * math.sqrt(self.hid_dim)
        x = x + self.pe(x)
        x = x.reshape(b, -1, T, D).permute(0, 3, 1, 2)

        for i in range(4):
            j = i * 2
            k = i * 3
            x1, x2, x = self.cell(x, self.dcc1[j], self.dcc1[j+1], self.trans[k], self.trans[k+1], self.trans[k+2],
                          self.diff_gcn[i])
            skip = self.skip_connect[i * 2](x1) + skip
            skip = self.skip_connect[i * 2 + 1](x2) + skip

        x = F.relu(skip)  # [64, 256, 170, 12]
        x = torch.max(x, dim=-1, keepdim=True)[0]

        x = F.relu(self.end_linear_1(x))
        x = self.end_linear_2(x)

        return x
