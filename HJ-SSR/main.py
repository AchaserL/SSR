import tensorflow as tf
import tensorflow.compat.v1 as tf
from include.Config import Config
from include.Model import build, training
from include.Test import *
from include.Load import *
from include.Utils import *
from include.str_boost_0310 import get_accuracy_by_structure as gabs
import warnings
import os
from include.pre_deal import *
import numba as nb
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 用CPU跑

warnings.filterwarnings("ignore")

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

tf.disable_eager_execution()
'''
Followed the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1))) # 实体总数

    ILL = loadfile(Config.ill, 2) # 数据集
    illL = len(ILL)
    np.random.shuffle(ILL) # 打乱
    train = np.array(ILL[:illL // 10 * Config.seed]) # 划分出训练集
    test = ILL[illL // 10 * Config.seed:] # 测试集
    test_r = loadfile(Config.ill_r, 2) # 带关系的测试集
    
    KG1 = loadfile(Config.kg1, 3) # 三元组集合
    KG2 = loadfile(Config.kg2, 3) # 三元组集合
    
    adj = get_weighted_adj(e, KG1 + KG2)
    
    # 建立模型
    output_prel_e, output_joint_e, output_r, loss_1, loss_2, head, tail = \
        build(Config.dim, Config.act_func, Config.gamma, Config.k, Config.language[0:2], e, train, KG1 + KG2)
    J = training(output_prel_e, output_joint_e, output_r, loss_1, loss_2, 0.001, Config.epochs, train, e, 
                 Config.k, Config.s, test, test_r, head, tail, adj)
    print('loss:', J)
