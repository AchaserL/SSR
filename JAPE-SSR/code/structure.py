from scipy import spatial
import numpy as np
import json


def get_accuracy_by_structure(vec, references_s, references_t, can_len, adj, weight):

    Lvec = np.array([vec[e] for e in references_s])
    Lent2mx = {e: i for (i, e) in enumerate(references_s)}
    mx2Lent = {i: e for (i, e) in enumerate(references_s)}

    Rvec = np.array([vec[e] for e in references_t])
    # Rent2mx = {e: i for (i, e) in enumerate(references_t)}
    mx2Rent = {i: e for (i, e) in enumerate(references_t)}

    sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')

    candidates = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        candidates.append(list(map(lambda x: mx2Rent[x], rank[:can_len])))

    candidates = np.array(candidates)

    adj = adj.tocsr()

    neighbours = []
    for i in range(Lvec.shape[0]):
        entId = mx2Lent[i]
        edge = []
        for neighbour in adj[entId].nonzero()[1]:
            if Lent2mx.get(neighbour, -1) != -1:
                edge.append(neighbour)
        neighbours.append(edge)

    accuracy = 0.0
    for i in range(Lvec.shape[0]):
        structured_enhance = {}
        for rankIdx1 in range(can_len):
            r_ent_id = candidates[i, rankIdx1]

            rankCnt = can_len - rankIdx1
            dist_rank_nei = 0
            for neighbour in neighbours[i]:
                for rankIdx2 in range(can_len):
                    l_ent_id = Lent2mx.get(neighbour, -1)
                    if l_ent_id == -1:
                        break
                    r_neighbour_id = candidates[l_ent_id, rankIdx2]
                    if adj[r_ent_id, r_neighbour_id]:
                        dist_rank_nei += can_len - rankIdx2
                        break
            structured_enhance.update(
                {r_ent_id: (rankCnt + weight * dist_rank_nei)}
            )
        ordered = sorted(structured_enhance, key=structured_enhance.get, reverse=True)

        accuracy += 1.0 if ordered[0] == mx2Rent[i] else 0.0
    accuracy = accuracy / Lvec.shape[0]

    print('\nL -> R Structure Enhanced Accuracy: %.2f%%' % (accuracy * 100), weight)

