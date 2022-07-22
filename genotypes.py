from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')  # 前面是类名，后面是字段名

PRIMITIVES = [
    # 'none',
    'skip_connect',
    'dcc_1',
    # 'dcc_2',
    'diff_gcn',
    'trans',
    's_trans',
    # 'cheb_gcn',
    # 'cnn',
    # 'att1',
    # 'att2',
    # 'lstm',
    # 'gru'
]
