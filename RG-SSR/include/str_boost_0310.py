import numpy as np
from scipy import spatial
import json


def get_accuracy_by_structure(candidates, neighbours, right_rank, can_len, adj, weight,
                              Lvec, Lent2mx, mx2Lent, mx2Rent):
    accuracy = 0.0
    # structure_dic = {}
    for i in range(Lvec.shape[0]):  # 遍历测试集中的左实体
        # if right_rank[i] >= can_len:  # 如果真实实体不在候选集中，直接下一个，不计accuracy
        #     continue
        # degree = len(neighbours[i])
        # a, b = structure_dic.get(degree, (0, 0))
        structured_enhance = {}  # 基于结构相似性的新的候选集
        if right_rank[i] >= can_len:
            continue
        for rank_idx1 in range(can_len):  # 对左实体候选集中的每个实体
            r_ent_id = candidates[i, rank_idx1]  # 右实体在图谱中的id

            dist_rank_ori = can_len - rank_idx1  # 只考虑距离的排名和
            dist_rank_nei = 0  # 邻居距离
            # relation_fun = 0   # 关系权重

            for neighbour in neighbours[i]:
                # me2Lent[i]从测试集enumerate到左实体id
                # relation_fun += adj[mx2Lent[i], neighbour]  # 边权和，越大越靠前
                for rank_idx2 in range(can_len):
                    l_ent_id = Lent2mx.get(neighbour, -1)
                    if l_ent_id == -1:  # 如果邻居不在测试集中，直接退出当前for，下一个邻居
                        break
                    r_neighbour_id = candidates[l_ent_id, rank_idx2]
                    if adj[r_ent_id, r_neighbour_id]:
                        dist_rank_nei += can_len - rank_idx2
                        break
            structured_enhance.update(
                {r_ent_id: (dist_rank_ori + weight * dist_rank_nei)})
        ordered = sorted(structured_enhance, key=structured_enhance.get, reverse=True)
        # if ordered[0] == mx2Rent[i] and candidates[i, 0] != mx2Rent[i]:  # 重排成功
        #     a = a + 1
        # if ordered[0] != mx2Rent[i] and candidates[i, 0] == mx2Rent[i]:  # 重排失败
        #     b = b + 1
        # structure_dic.update({degree: (a, b)})
        accuracy += 1.0 if ordered[0] == mx2Rent[i] else 0.0
    accuracy /= Lvec.shape[0]
    # with open('structure_dic_epoch300_zh_en.json', 'w') as f:
    #     json.dump(structure_dic, f, ensure_ascii=False)
    # print('\nL -> R Structure enhance accuracy: {:.2%}'.format(accuracy), weight)
    print('{:.2%}'.format(accuracy), weight)
