import numpy as np
from scipy import spatial
import json


def get_accuracy_by_structure(vec, test_pair, can_len, adj, weight):
    """
        Args:
            vec:  Entities Embeddings
            test_pair: Testing Data
            can_len: The length of candidate set
            adj: Adjacent Matrix
            weight: edge weight

        Returns: some data
        """
    Lvec = np.array([vec[e1] for e1, _ in test_pair])  # 测试集左侧实体嵌入
    Lent2mx = {e1: i for i, (e1, _) in enumerate(test_pair)}  # 左侧实体id对应的测试集中的id
    mx2Lent = {i: e1 for i, (e1, _) in enumerate(test_pair)}
    Rvec = np.array([vec[e2] for _, e2 in test_pair])  # 测试集右侧实体嵌入
    mx2Rent = {i: e2 for i, (_, e2) in enumerate(test_pair)}  # 测试及集中的id对应右侧实体id
    # sim里都是测试集中出现过的节点，里面的id都是测试集enumerate id
    sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')

    # right_rank = []  # 测试集右侧对齐实体依距离的排名
    candidates = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()  # 经过argsort得到的是id
        # right_rank.append(np.where(rank == i)[0][0])
        candidates.append(list(map(lambda x: mx2Rent[x], rank[:can_len])))  # candidates里是前can_len个右实体在图谱中的id
    candidates = np.array(candidates)  # list -> array
    adj = adj.tocsr()  # csr格式索引
    # local structure
    # neighbour[i]是左实体id
    neighbours = []
    for i in range(Lvec.shape[0]):
        ent_id = mx2Lent[i]  # 左实体id
        edge = []  # 保存其邻居
        for neighbour in adj[ent_id].nonzero()[1]:  # 遍历邻接矩阵中左实体ent_id的邻居节点
            if Lent2mx.get(neighbour, -1) != -1:  # 只添加在测试集中出现过的邻居
                edge.append(neighbour)
        neighbours.append(edge)

    accuracy = 0.0
    # degree_dic = {}
    # structure_dic = {}  # 结构字典，key: 度，value: (准确度提升总个数，准确度下降总个数)
    for i in range(Lvec.shape[0]):  # 遍历测试集中的左实体
        # if right_rank[i] >= can_len:  # 如果真实实体不在候选集中，直接下一个，不计accuracy
        #     continue
        # a, b = structure_dic.get(degree, (0, 0))
        # degree_dic.update({degree: degree_dic.get(degree, 0) + 1})
        structured_enhance = {}  # 基于结构相似性的新的候选集
        for rank_idx1 in range(can_len):  # 对左实体候选集中的每个实体
            r_ent_id = candidates[i, rank_idx1]  # 右实体在图谱中的id

            dist_rank_ori = can_len - rank_idx1  # 只考虑距离的排名和
            dist_rank_nei = 0  # 邻居距离

            for neighbour in neighbours[i]:
                # me2Lent[i]从测试集enumerate到左实体id
                for rank_idx2 in range(can_len):
                    l_ent_id = Lent2mx.get(neighbour, -1)
                    if l_ent_id == -1:  # 如果邻居不在测试集中，直接退出当前for，下一个邻居
                        break
                    r_neighbour_id = candidates[l_ent_id, rank_idx2]
                    if adj[r_ent_id, r_neighbour_id]:
                        dist_rank_nei += can_len - rank_idx2
                        break
            structured_enhance.update(
                {r_ent_id: (dist_rank_ori + weight * dist_rank_nei)}
            )
        ordered = sorted(structured_enhance, key=structured_enhance.get, reverse=True)
        # if ordered[0] == mx2Rent[i] and candidates[i,0] != mx2Rent[i]:
        #     a = a + 1
        # if ordered[0] != mx2Rent[i] and candidates[i,0] == mx2Rent[i]:
        #     b = b + 1
        # structure_dic.update({degree: (a, b)})
        accuracy += 1.0 if ordered[0] == mx2Rent[i] else 0.0
    # with open('structure_dic.json', 'w', encoding='utf-8') as f:
    #     json.dump(structure_dic, f, ensure_ascii=False)
    accuracy /= Lvec.shape[0]
    print('\nL -> R Structure enhance accuracy: {:.2%}'.format(accuracy), weight)
    # print('{:.2%}'.format(accuracy), weight)
