import numpy as np
from scipy import spatial


def get_accuracy_by_structure(lvec, rvec, l_id, r_id, can_len, adj, weight):
    """
    根据结构重排
    Args:
        lvec: 测试集左图谱向量
        rvec: 测试集右图谱向量
        l_id: 测试集左图谱id
        r_id: 测试集右图谱id
        can_len: 候选集重排长度
        adj: 邻接矩阵
        weight: 权重

    Returns:
    """
    Lent2mx = {e1: i for i, e1 in enumerate(l_id)}
    mx2Lent = {i: e1 for i, e1 in enumerate(l_id)}
    mx2Rent = {i: e2 for i, e2 in enumerate(r_id)}

    sim = spatial.distance.cdist(lvec, rvec, metric='cityblock')

    candidates = []
    for i in range(lvec.shape[0]):
        rank = sim[i, :].argsort()
        candidates.append(list(map(lambda x: mx2Rent[x], rank[:can_len])))
    candidates = np.array(candidates)

    adj = adj.tocsr()
    neighbors = []
    for i in range(lvec.shape[0]):
        ent_id = mx2Lent[i]
        edge = []
        for neighbor in adj[ent_id].nonzero()[1]:
            if Lent2mx.get(neighbor, -1) != -1:
                edge.append(neighbor)
        neighbors.append(edge)

    accuracy = 0.0
    for i in range(lvec.shape[0]):
        structured_enhance = {}
        for rank_idx1 in range(can_len):
            r_ent_id = candidates[i, rank_idx1]

            dist_rank_ori = can_len - rank_idx1
            dist_rank_nei = 0
            for neighbor in neighbors[i]:
                for rank_idx2 in range(can_len):
                    l_ent_id = Lent2mx.get(neighbor, -1)
                    if l_ent_id == -1:
                        break
                    r_neighbor_id = candidates[l_ent_id, rank_idx2]
                    if adj[r_ent_id, r_neighbor_id]:
                        dist_rank_nei += can_len - rank_idx2
                        break
            structured_enhance.update({r_ent_id: (dist_rank_ori + weight * dist_rank_nei)})
        ordered = sorted(structured_enhance, key=structured_enhance.get, reverse=True)
        accuracy += 1.0 if ordered[0] == mx2Rent[i] else 0.0
    accuracy /= lvec.shape[0]
    print('\nL -> R Structure enhance accuracy: {:.2%}'.format(accuracy), weight)
