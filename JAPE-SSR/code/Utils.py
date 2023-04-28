import numpy as np
import scipy.sparse as sp
import math


def get_adj(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
    
    row  = []
    col  = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))
