import numpy as np
from scipy import spatial


def pre_deal(vec, test_pair, can_len, adj):
    """
    Args:
        vec:  Entities Embeddings
        test_pair: Testing Data
        can_len: The length of candidate set
        adj: Adjacent Matrix

    Returns: some data
    """
    Lvec = np.array([vec[e1] for e1, _ in test_pair])  # 测试集左侧实体嵌入
    Lent2mx = {e1: i for i, (e1, _) in enumerate(test_pair)}  # 左侧实体id对应的测试集中的id
    mx2Lent = {i: e1 for i, (e1, _) in enumerate(test_pair)}
    Rvec = np.array([vec[e2] for _, e2 in test_pair])  # 测试集右侧实体嵌入
    mx2Rent = {i: e2 for i, (_, e2) in enumerate(test_pair)}  # 测试及集中的id对应右侧实体id
    # sim里都是测试集中出现过的节点，里面的id都是测试集enumerate id
    sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')

    right_rank = []  # 测试集右侧对齐实体依距离的排名
    candidates = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()  # 经过argsort得到的是id
        right_rank.append(np.where(rank == i)[0][0])
        candidates.append(list(map(lambda x: mx2Rent[x], rank[:can_len])))  # candidates里是前can_len个右实体在图谱中的id
    candidates = np.array(candidates)  # list -> array
    # adj = adj.tocsr()  # csr格式索引
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
    print('Preprocess done!')
    return candidates, right_rank, neighbours, Lvec, Lent2mx, mx2Lent, mx2Rent
