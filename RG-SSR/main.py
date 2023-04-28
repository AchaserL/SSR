import tensorflow as tf
import tensorflow.compat.v1 as tf
from include.Config import Config
from include.Model import build, training
from include.Test import get_hits
from include.Load import *
from include.Utils import *
from include.str_boost_0310 import get_accuracy_by_structure as gabs
from include.test_degree import *
import warnings
import os
from include.pre_deal import *
import numba as nb
import time
import json


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings("ignore")

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

tf.disable_eager_execution()

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)  # ILL是list
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]

    # KG是三元组列表
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    print('getting matrix...')
    adj = get_adj(e, KG1 + KG2)
    adj = adj.tocsr()
    print('adj matrix done.')

    # # 训练集对齐实体id字典
    # train_dic_lr = {}
    # for row in range(train.shape[0]):
    #     l_id, r_id = train[row, 0], train[row, 1]
    #     train_dic_lr.update({l_id: r_id})

    # print('getting degree')
    # degree_ts = get_test_pairs_degree(adj, test)
    # degree_tr = get_test_pairs_degree(adj, ILL[:illL // 10 * Config.seed])\
    #
    # with open('train_pairs_degree.json', 'w') as f:
    #     json.dump(degree_tr, f)
    #
    # with open('test_pairs_degree.json', 'w') as f:
    #     json.dump(degree_ts, f)

    # print('degree done!')

    output_layer, loss = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k,
        Config.language[0:2], e, train, KG1 + KG2)
    vec, J = training(output_layer, loss, 0.001, Config.epochs, train, e, Config.k, test, adj)
    type(vec)
    print('loss:', J)
    print('Result:')
    get_hits(vec, test)

    candidates, right_rank, neighbours, \
    Lvec, Lent2mx, mx2Lent, mx2Rent \
        = pre_deal(vec, test, Config.can_len, adj)

    for weight in range(0, 55, 5):
        weight /= 100
        gabs(candidates, neighbours, right_rank, Config.can_len, adj,
             weight, Lvec, Lent2mx, mx2Lent, mx2Rent)
