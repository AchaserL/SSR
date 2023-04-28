import numpy as np
from scipy import spatial
import json


def get_accuracy_by_structure(vec, test_pair, can_len, adj, weight):
    """结合结构相似的跨语言实体对应查找
    Parameters:
        adj       - 稀疏矩阵，两个图的邻接矩阵，未经过正则化
        can_len   - 候选集长度
    """
    
    Lvec    = np.array([vec[e1] for e1, _ in test_pair])      # 测试集左侧实体的训练后的嵌入
    Lent2mx = {e1: i for i, (e1, _) in enumerate(test_pair)}   # 左侧实体id对应测试集中的id
    mx2Lent = {i: e1 for i, (e1, _) in enumerate(test_pair)}
    
    Rvec    = np.array([vec[e2] for _, e2 in test_pair])      # 测试集右侧实体的训练后的嵌入
    # Rent2mx = {e2: i for i, (_, e2) in enumerate(test_pair)}   # 右侧实体id对应测试集中的id
    mx2Rent = {i: e2 for i, (_, e2) in enumerate(test_pair)}
    
    # sim中都是测试集id
    # 只考虑测试集出现过的节点
    sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    
    # L -> R
    # 找到每个节点的跨语言对应候选集
    candidates = []  # candidates[i][_]为测试集中左节点i的KG2中的对应节点的图谱id
    for i in range(Lvec.shape[0]):
        
        rank = sim[i, :].argsort()
        candidates.append(list(map(lambda x: mx2Rent[x], rank[:can_len])))
    
    candidates = np.array(candidates)  # list -> array

    adj = adj.tocsr()  # csr格式支持索引
    
    # 找到每个节点的邻接节点，即查找子图
    neighbours = []   # neigbours[i]表示测试集中左节点i的KG1中的邻接节点的图谱id
    for i in range(Lvec.shape[0]):
        entId = mx2Lent[i]
        edge = []
        for neighbour in adj[entId].nonzero()[1]:
            if Lent2mx.get(neighbour, -1) != -1:  # 只添加测试集中出现过的邻居
                edge.append(neighbour)
        neighbours.append(edge)

    # 在候选集中找到与左图结构相似的图结构，增强匹配
    structure_dic = {}
    accuracy = 0.0
    for i in range(Lvec.shape[0]):  # 对测试集左图每个节点
        # print('\r' + str(i), end='')
        structured_enhance = {}     # 构建其结构增强的候选集
        for rankIdx1 in range(can_len):  # 测试集左节点的候选集r_ent_id
            # 对测试集左节点的候选集中的每个图谱id(右图谱id) r_ent_id
            # 需要对其遍历左节点的所有邻居的候选集
            # 找到在候选集中也满足左图中结构的右图谱图谱id r_ent_id
            # 如果不唯一，综合排名靠前的排在前面
            r_ent_id = candidates[i, rankIdx1]
            rankCnt = can_len - rankIdx1
            dist_rank_nei = 0
            for neighbour in neighbours[i]:  # neighbour为左实体在KG1中的邻居的图谱id(右图实体id)
                for rankIdx2 in range(can_len):  # 邻居的候选集candidates[l_ent_id]
                    l_ent_id = Lent2mx.get(neighbour, -1) 
                    if l_ent_id == -1:  # 如果neighbour不在测试集中，则不考虑当前neighbour
                        break
                    r_neighbour_id = candidates[l_ent_id, rankIdx2]
                    if adj[r_ent_id, r_neighbour_id]:  # 如果r_ent_id和r_neighbour_id在右图中有边
                        dist_rank_nei += can_len - rankIdx2  # 排名计数累加
                        break
            structured_enhance.update(
                {r_ent_id: (rankCnt + weight * dist_rank_nei)}
            )
        ordered = sorted(structured_enhance, key=structured_enhance.get, reverse=True)  # 按值排序
        # 得到结构增强的跨语言对应预测
        # if ordered[0] == mx2Rent[i] and candidates[i, 0] != mx2Rent[i]:
        #     a = a + 1
        # if ordered[0] != mx2Rent[i] and candidates[i, 0] == mx2Rent[i]:
        #     b = b + 1
        # structure_dic.update({degree: (a, b)})
        # print('to here!')
        accuracy += 1.0 if ordered[0] == mx2Rent[i] else 0.0
    accuracy = accuracy / Lvec.shape[0]
    # with open('structure_dic.json', 'w') as f:
    #     json.dump(structure_dic, f, ensure_ascii=False)
    print('\nL -> R Structure enhance accuracy: {:.2%}'.format(accuracy), weight)
    # with open('results.json', 'a+') as f:
    #     json.dump((accuracy, weight), f, ensure_ascii=False)
