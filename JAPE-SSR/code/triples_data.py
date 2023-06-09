class Triples_Data:
    def __init__(self, triples):
        self.ent_num, self.rel_num = 0, 0
        self.train_triples = triples
        self.train_triples_set = set(triples)
        self.train_triples_num = len(triples)
        self.ents, self.rels = set(), set()
        self.ents_list, self.tails_list, self.heads_list, self.rels_list = list(), list(), list(), list()
        self.rel2htth = dict()
        self.r_hs_train, self.r_ts_train = dict(), dict()
        self.__init_data()

    def __init_data(self):
        heads = set([triple[0] for triple in self.train_triples]) # 头实体id集
        tails = set([triple[2] for triple in self.train_triples]) # 尾实体id集
        self.rels = set([triple[1] for triple in self.train_triples]) # 关系id集
        self.ents = heads | tails # 实体集：头实体集 并 尾实体集
        self.ents_list = list(self.ents) # 实体列表
        self.tails_list = list(tails)    # 尾实体列表
        self.heads_list = list(heads)    # 头实体列表
        self.rels_list = list(self.rels) # 关系列表
        self.ent_num = len(self.ents)    # 实体数
        self.rel_num = len(self.rels)    # 关系数
        print('ents: %d + %d = %d' % (len(heads), len(tails), self.ent_num))
        # 构造r_hs_train和r_ts_train，{关系：实体列表} 字典
        for (h, r, t) in self.train_triples:
            self.__add_dict_kv(self.r_hs_train, r, h) 
            self.__add_dict_kv(self.r_ts_train, r, t)
        for r in self.r_hs_train.keys():
            self.r_hs_train[r] = list(self.ents - self.r_hs_train[r])
            self.r_ts_train[r] = list(self.ents - self.r_ts_train[r])
            # self.r_hs_train[r] = list(self.r_hs_train[r])
            # self.r_ts_train[r] = list(self.r_ts_train[r])
        self.__count_ht_th()

    def __count_ht_th(self):
        dic_r_h, dic_r_t, dic_r_num = dict(), dict(), dict()
        for triple in self.train_triples:
            self.__add_dict_num(dic_r_num, triple[1])
            self.__add_dict_kv(dic_r_h, triple[1], triple[0])
            self.__add_dict_kv(dic_r_t, triple[1], triple[2])
        for r in dic_r_h.keys():
            h_num = len(dic_r_h[r])
            t_num = len(dic_r_t[r])
            triple_num = dic_r_num[r]
            self.rel2htth[r] = (round(triple_num / h_num, 5), round(triple_num / t_num, 5))

    def __add_dict_kvs(self, dic, k, vset):
        vs = dic.get(k, set())
        vs = vs | vset
        dic[k] = vs

    def __add_dict_kv(self, dic, k, v):
        vs = dic.get(k, set())
        vs.add(v)
        dic[k] = vs

    def __add_dict_num(self, dic, k):
        if dic.get(k) is None:
            dic[k] = 1
        else:
            dic[k] += 1

    def exist(self, r, h, t):
        return (h, r, t) in self.train_triples_set
